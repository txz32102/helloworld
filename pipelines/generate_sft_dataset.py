import os
import json
import glob
from .utils import get_openai_client, generate_llm_response
from .tools.registry import TOOL_SCHEMAS, AVAILABLE_TOOLS

class SFTTrajectoryGenerator:
    def __init__(self, data_dir: str, teacher_model_id: str, output_jsonl: str, client=None):
        """
        Standalone pipeline strictly for generating SFT training trajectories.
        Does not pollute the test-time generation pipeline.
        """
        self.data_dir = data_dir
        self.teacher_model_id = teacher_model_id
        self.output_jsonl = output_jsonl
        self.client = client if client else get_openai_client()

    def _read_file_safe(self, filepath: str) -> str:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    def _execute_teacher_loop(self, messages: list, case_data: dict) -> list:
        """Runs the ReAct loop and returns the pristine trajectory list."""
        trajectory = []
        
        print(f"    [*] Starting Teacher LLM Tool Loop ({self.teacher_model_id})...")
        
        for turn in range(15):
            print(f"    [*] Turn {turn + 1}: Generating reasoning step...")
            
            response_data = generate_llm_response(
                client=self.client,
                model=self.teacher_model_id,
                messages=messages,
                stream=False, 
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2 
            )

            # --- THE CRITICAL UPDATE ---
            # Safely use .get() in case the fields are missing
            collected_content = response_data.get("content", "")
            tool_calls = response_data.get("tool_calls", [])
            
            # Extract the thinking process (Check your specific API documentation, 
            # but 'reasoning_content' is the industry standard for OpenAI-compatible endpoints)
            thinking_content = response_data.get("reasoning_content", "")

            # If no tools are called, the model is outputting the final report
            if not tool_calls:
                final_turn = {
                    "role": "assistant", 
                    "reasoning_content": thinking_content, # Store final thoughts
                    "content": collected_content
                }
                trajectory.append(final_turn)
                break
                
            # Append Teacher's thought, reasoning, and tool_calls to trajectory
            assistant_turn = {
                "role": "assistant",
                "reasoning_content": thinking_content, # <--- THIS IS THE GOLD FOR SFT
                "content": collected_content if collected_content else "",
                "tool_calls": tool_calls
            }
            
            messages.append(assistant_turn)
            trajectory.append(assistant_turn)
            
            # Execute Tools (Unchanged)
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_args_str = tool_call["function"]["arguments"]
                function_to_call = AVAILABLE_TOOLS.get(function_name)
                
                try:
                    function_args = json.loads(function_args_str)
                    function_args["case_data"] = case_data 
                    
                    if function_to_call:
                        function_response = function_to_call(**function_args)
                    else:
                        function_response = f"Error: Tool {function_name} not found."
                except Exception as e:
                    function_response = f"Tool execution error: {str(e)}"
                
                # Append Observation to trajectory
                tool_turn = {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                }
                messages.append(tool_turn)
                trajectory.append(tool_turn)
                
        return trajectory

    def generate_trajectory_for_case(self, case_dir: str):
        case_id = os.path.basename(case_dir)
        atoms_path = os.path.join(case_dir, f"{case_id}_atoms.json")
        gt_path = os.path.join(case_dir, f"{case_id}_gt.md")
        
        if not os.path.exists(atoms_path) or not os.path.exists(gt_path):
            print(f"[!] Skipping {case_id}: Missing atoms or GT file.")
            return

        print(f"\nProcessing Case ID: {case_id} for SFT Trajectory")
        
        # Load the unstructured atoms and the perfect Ground Truth
        with open(atoms_path, 'r', encoding='utf-8') as f:
            case_data = json.load(f)
        ground_truth_md = self._read_file_safe(gt_path)

        # Build the Teacher Prompt (Notice how we inject the GT here)
        system_instruction = (
            "You are an expert AI teacher generating training data for a clinical reasoning agent. "
            "Your task is to demonstrate the PERFECT step-by-step reasoning and tool-use required "
            "to bridge the gap between unstructured clinical notes and a final academic report.\n\n"
            "CRITICAL: I will provide you with the target 'Ground Truth' report. "
            "You must generate the thoughts and execute the specific `search_pubmed` and `fetch_ama_citation` "
            "tool calls that would logically lead to creating THIS exact report. "
            "Once you have gathered the necessary citations via tools, output the exact Ground Truth report as your final step."
        )
        
        user_payload = f"### UNSTRUCTURED ATOMS ###\n{json.dumps(case_data, indent=2)}\n\n### TARGET GROUND TRUTH REPORT ###\n{ground_truth_md}"

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_payload}
        ]

        # Get the pristine step-by-step sequence
        trajectory = self._execute_teacher_loop(messages, case_data)

        # Save to JSONL for SFT Training
        sft_instance = {
            "case_id": case_id,
            "system": "You are an expert medical researcher orchestrating a clinical case report...", # The actual prompt the student will see
            "input_atoms": case_data,
            "trajectory": trajectory # The teacher's step-by-step logic
        }

        with open(self.output_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(sft_instance) + "\n")
            
        print(f"[+] Successfully saved trajectory for {case_id} ({len(trajectory)} steps)")

    def run(self):
        # Finds all case directories (e.g., 41674886)
        case_dirs = [d for d in glob.glob(os.path.join(self.data_dir, "*")) if os.path.isdir(d)]
        
        print(f"Starting SFT Trajectory generation for {len(case_dirs)} cases...")
        for case_dir in case_dirs:
            self.generate_trajectory_for_case(case_dir)
        print("\nSFT Dataset generation complete.")