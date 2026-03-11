import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# --- CONFIGURATION ---
DATA_ROOT = "../data/download_files"
PROGRESS_FILE = "../log/progress.txt"

os.makedirs("../log", exist_ok=True)

# Mount the root data folder
app.mount("/data_root", StaticFiles(directory=DATA_ROOT), name="data_root")

class ActionModel(BaseModel):
    folder_id: str
    action: str

def get_progress():
    """Reads progress.txt and returns a dict mapping folder_id -> status (accept/reject)"""
    if not os.path.exists(PROGRESS_FILE): 
        return {}
    prog = {}
    with open(PROGRESS_FILE, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                prog[parts[0]] = parts[1]
    return prog

# 1. SERVE THE STATIC HTML SHELL (Only happens once)
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE

# 2. API ENDPOINT TO GET FOLDER DATA
# 2. API ENDPOINT TO GET FOLDER DATA
@app.get("/api/data")
async def get_data(folder_id: str = None, auto_advance: bool = False):
    progress_data = get_progress()
    
    # Only include folders that actually contain a PDF file
    valid_folders = []
    for f in os.listdir(DATA_ROOT):
        folder_path = os.path.join(DATA_ROOT, f)
        if os.path.isdir(folder_path):
            if any(file.lower().endswith('.pdf') for file in os.listdir(folder_path)):
                valid_folders.append(f)
                
    all_folders = sorted(valid_folders)
    current_id = folder_id
    
    # THE FIX: Only auto-advance if the user just clicked Accept/Reject
    if auto_advance and current_id in all_folders:
        current_index = all_folders.index(current_id)
        # Search forward for the next unprocessed one
        next_id = next((id for id in all_folders[current_index + 1:] if id not in progress_data), None)
        # Loop back to the beginning if we hit the end
        if not next_id:
            next_id = next((id for id in all_folders if id not in progress_data), None)
        current_id = next_id

    # If no ID provided (initial load) or invalid, find the first un-annotated
    if not current_id or current_id not in all_folders:
        current_id = next((id for id in all_folders if id not in progress_data), None)
            
    # Fallback to the last one if everything is complete
    if not current_id and all_folders:
        current_id = all_folders[-1] 

    folder_list = [{"id": f, "status": progress_data.get(f, "")} for f in all_folders]
    
    files = []
    if current_id:
        folder_path = os.path.join(DATA_ROOT, current_id)
        files = sorted(os.listdir(folder_path))
        
    pdf_file = next((f for f in files if f.endswith(".pdf")), "")
    json_file = next((f for f in files if f.endswith(".json")), None) or \
                next((f for f in files if f.endswith(".xml")), "")

    return {
        "current_id": current_id,
        "folders": folder_list,
        "files": files,
        "stats": {
            "accepted": sum(1 for v in progress_data.values() if v == "accept"),
            "rejected": sum(1 for v in progress_data.values() if v == "reject"),
            "total": len(all_folders)
        },
        "initial_pdf": f"/data_root/{current_id}/{pdf_file}" if pdf_file else "",
        "initial_json": f"/data_root/{current_id}/{json_file}" if json_file else ""
    }

# 3. API ENDPOINT TO SUBMIT ACTIONS
@app.post("/api/submit")
async def submit_folder(data: ActionModel):
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{data.folder_id},{data.action}\n")
    return {"status": "success"}


# --- REFRESH-FREE SPA TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Annotator</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f8fafc;
            --surface: #ffffff;
            --border: #cbd5e1;
            --text-main: #334155;
            --text-muted: #64748b;
            --brand: #2563eb;
            --resizer-color: #94a3b8;
        }
        
        body, html { height: 100%; margin: 0; overflow: hidden; font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg-color); color: var(--text-main); }
        
        /* Header */
        .header { height: 64px; background: var(--surface); border-bottom: 1px solid var(--border); display: flex; align-items: center; padding: 0 24px; justify-content: space-between; flex-shrink: 0; }
        .header-title { font-size: 1.1rem; font-weight: 600; margin: 0; display: flex; align-items: center; gap: 15px; }
        .stats { font-size: 0.85rem; color: var(--text-muted); font-weight: 400; }
        
        /* Main Layout */
        .main-container { display: flex; height: calc(100% - 64px); width: 100vw; flex-direction: row; overflow: hidden; }
        
        /* Pane Base Styles */
        .pane { display: flex; flex-direction: column; overflow: hidden; background: var(--surface); }
        .pane-content { overflow-y: auto; flex: 1; }
        
        /* Specific Panes */
        #pane-folders { width: 140px; min-width: 80px; }
        #pane-files { width: 180px; min-width: 100px; background: var(--bg-color); padding: 8px; }
        #pane-viewer { flex: 1; min-width: 200px; background: #94a3b8; position: relative; }
        #pane-meta { width: 280px; min-width: 150px; padding: 10px; }
        
        /* Resizer Bars */
        .resizer { width: 6px; background: var(--border); cursor: col-resize; flex-shrink: 0; display: flex; justify-content: center; align-items: center; transition: background 0.2s; }
        .resizer:hover, .resizer:active { background: var(--resizer-color); }
        .resizer::after { content: "⋮"; color: #fff; font-size: 12px; font-weight: bold; pointer-events: none; opacity: 0.6; }

        /* Inner Content Styles */
        .list-group-item { border: none; border-bottom: 1px solid var(--border); font-size: 0.8rem; padding: 10px 12px; color: var(--text-main); word-break: break-all; text-decoration: none;}
        .list-group-item:hover { background-color: #f1f5f9; }
        .list-group-item.active { background-color: var(--brand); color: white; border-color: var(--brand); }
        
        .pane-title { font-size: 0.65rem; text-transform: uppercase; font-weight: 700; color: var(--text-muted); margin-bottom: 8px; letter-spacing: 0.5px; flex-shrink: 0; }
        .file-item { padding: 6px 8px; cursor: pointer; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; margin-bottom: 6px; font-size: 0.75rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; transition: all 0.2s; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
        .file-item:hover { border-color: #cbd5e1; transform: translateY(-1px); box-shadow: 0 4px 6px rgba(0,0,0,0.04); }
        
        pre { background: var(--bg-color); border: 1px solid var(--border); padding: 10px; border-radius: 8px; font-size: 0.75rem; flex: 1; overflow: auto; color: #0f172a; margin: 0; white-space: pre-wrap; word-wrap: break-word; height: 100%; }
        
        .embed-container { width: 100%; height: 100%; }
        embed { width: 100%; height: 100%; border: none; }
        .img-preview { max-width: 100%; max-height: 100%; object-fit: contain; margin: auto; display: block; background: #fff; }
        .no-pdf-msg { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #fff; font-weight: 500; text-shadow: 0 1px 2px rgba(0,0,0,0.5); }
    </style>
</head>
<body>
    <div class="header">
        <h1 class="header-title">
            Annotator 
            <span id="current-badge" class="badge bg-light text-dark border px-3 py-2 rounded-pill fw-normal">Current: Loading...</span>
            <div id="stats-text" class="stats">Loading Stats...</div>
        </h1>
        
        <div class="d-flex m-0 gap-2">
            <button onclick="submitAction('reject')" class="btn btn-outline-danger btn-sm px-4 fw-bold rounded-pill">✗ REJECT</button>
            <button onclick="submitAction('accept')" class="btn btn-success btn-sm px-4 fw-bold rounded-pill shadow-sm">✓ ACCEPT & NEXT</button>
        </div>
    </div>

    <div class="main-container">
        <div class="pane" id="pane-folders">
            <div class="pane-content list-group list-group-flush" id="folder-list-container">
                </div>
        </div>
        <div class="resizer" id="resizer-folders"></div>

        <div class="pane" id="pane-files">
            <div class="pane-title">Directory Files</div>
            <div class="pane-content" id="file-list-container">
                </div>
        </div>
        <div class="resizer" id="resizer-files"></div>

        <div class="pane" id="pane-viewer">
            <div id="no-pdf-text" class="no-pdf-msg" style="display:none;">No image or PDF selected</div>
            <div class="embed-container" id="embed-wrapper">
                <embed id="pdf-frame" src="" type="application/pdf" style="display:none;"></embed>
            </div>
            <img id="img-frame" class="img-preview" style="display:none;">
        </div>
        <div class="resizer" id="resizer-meta"></div>

        <div class="pane" id="pane-meta">
            <div class="pane-title">Raw Data View</div>
            <div class="pane-content">
                <pre id="meta-view">Loading metadata...</pre>
            </div>
        </div>
    </div>

    <script>
        let currentFolderId = null;
        
        // --- SPA DATA FETCHING ---
        async function loadAppState(folderId = null, autoAdvance = false) {
            let url = '/api/data';
            if (folderId) url += `?folder_id=${folderId}`;
            if (autoAdvance) url += `&auto_advance=true`;
            
            try {
                const response = await fetch(url);
                const data = await response.json();
                
                if (!data.current_id) {
                    document.body.innerHTML = "<h1 style='padding:2rem'>No folders found in DATA_ROOT.</h1>";
                    return;
                }

                currentFolderId = data.current_id;
                document.title = `Annotator - ${currentFolderId}`;
                
                // Update Header
                document.getElementById('current-badge').innerText = `Current: ${currentFolderId}`;
                document.getElementById('stats-text').innerHTML = `${data.stats.accepted} Accepted &nbsp;|&nbsp; ${data.stats.rejected} Rejected &nbsp;|&nbsp; ${data.stats.total} Total`;
                
                // THE FIX: SAVE the current scroll position before touching the HTML
                const folderContainer = document.getElementById('folder-list-container');
                const savedScrollPosition = folderContainer.scrollTop;

                // Update Folders
                const folderHtml = data.folders.map(f => {
                    let badge = '';
                    if (f.status === 'accept') badge = '<span class="badge bg-success float-end rounded-pill" style="font-size:0.6rem;">✓</span>';
                    else if (f.status === 'reject') badge = '<span class="badge bg-danger float-end rounded-pill" style="font-size:0.6rem;">✗</span>';
                    
                    let activeClass = f.id === currentFolderId ? 'active' : '';
                    return `<a href="#" onclick="loadAppState('${f.id}'); return false;" class="list-group-item ${activeClass}">${f.id} ${badge}</a>`;
                }).join('');
                
                folderContainer.innerHTML = folderHtml;
                
                // THE FIX: RESTORE the scroll position immediately
                folderContainer.scrollTop = savedScrollPosition;
                
                // Update Files
                const filesHtml = data.files.map(f => {
                    let icon = "📄";
                    let lowerF = f.toLowerCase();
                    if (lowerF.endsWith('.jpg') || lowerF.endsWith('.png') || lowerF.endsWith('.jpeg')) icon = "🖼️";
                    if (lowerF.endsWith('.pdf')) icon = "📕";
                    return `<div class="file-item" onclick="viewFile('${f}')" title="${f}"><span class="me-1">${icon}</span>${f}</div>`;
                }).join('');
                document.getElementById('file-list-container').innerHTML = filesHtml;

                // Load Initial Files into Viewers
                const pdfFrame = document.getElementById('pdf-frame');
                const imgFrame = document.getElementById('img-frame');
                const noPdfText = document.getElementById('no-pdf-text');
                const metaView = document.getElementById('meta-view');

                imgFrame.style.display = 'none';
                
                if (data.initial_pdf) {
                    pdfFrame.style.display = 'block';
                    noPdfText.style.display = 'none';
                    pdfFrame.src = data.initial_pdf;
                } else {
                    pdfFrame.style.display = 'none';
                    noPdfText.style.display = 'block';
                }

                if (data.initial_json) {
                    fetch(data.initial_json)
                        .then(r => r.text())
                        .then(t => metaView.innerText = t)
                        .catch(() => metaView.innerText = "Failed to load metadata.");
                } else {
                    metaView.innerText = "No JSON or XML found in directory.";
                }

                // Ensure the active folder is scrolled into view smoothly if it moved out of bounds
                const activeFolderEl = document.querySelector('.list-group-item.active');
                if (activeFolderEl) activeFolderEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

            } catch (error) {
                console.error("Error loading state:", error);
            }
        }

        // --- SUBMISSION LOGIC ---
        async function submitAction(action) {
            if (!currentFolderId) return;
            
            try {
                await fetch('/api/submit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder_id: currentFolderId, action: action })
                });
                
                // THE FIX: Tell loadAppState that we want to auto-advance
                loadAppState(currentFolderId, true); 
            } catch (error) {
                console.error("Error submitting:", error);
                alert("Failed to submit. Check connection.");
            }
        }

        // --- FILE VIEWING LOGIC ---
        async function viewFile(fileName) {
            const url = `/data_root/${currentFolderId}/${fileName}`;
            const lowerName = fileName.toLowerCase();
            const pdfFrame = document.getElementById('pdf-frame');
            const imgFrame = document.getElementById('img-frame');
            const noPdfText = document.getElementById('no-pdf-text');
            const metaView = document.getElementById('meta-view');
            
            if (lowerName.endsWith('.pdf')) {
                imgFrame.style.display = 'none';
                noPdfText.style.display = 'none';
                pdfFrame.style.display = 'block';
                pdfFrame.src = url;
            } else if (lowerName.endsWith('.jpg') || lowerName.endsWith('.jpeg') || lowerName.endsWith('.png')) {
                pdfFrame.style.display = 'none';
                noPdfText.style.display = 'none';
                imgFrame.style.display = 'block';
                imgFrame.src = url;
            } else {
                try {
                    const res = await fetch(url);
                    const text = await res.text();
                    metaView.innerText = text;
                } catch(e) {
                    metaView.innerText = "Error loading file: " + e;
                }
            }
        }

        // --- DRAG TO RESIZE LOGIC ---
        function initResizable(resizerId, targetPaneId, isRightSide) {
            const resizer = document.getElementById(resizerId);
            const targetPane = document.getElementById(targetPaneId);
            const viewerPane = document.getElementById('pane-viewer'); // Grab the viewer pane
            let startX, startWidth;

            resizer.addEventListener('mousedown', function(e) {
                startX = e.clientX;
                startWidth = targetPane.getBoundingClientRect().width;
                document.documentElement.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';

                // THE FIX: Disable pointer events on the viewer so the <embed> doesn't steal the mouse
                if (viewerPane) viewerPane.style.pointerEvents = 'none';

                const doDrag = function(e) {
                    if (isRightSide) {
                        targetPane.style.width = (startWidth - (e.clientX - startX)) + 'px';
                    } else {
                        targetPane.style.width = (startWidth + (e.clientX - startX)) + 'px';
                    }
                };

                const stopDrag = function() {
                    document.documentElement.style.cursor = '';
                    document.body.style.userSelect = '';
                    
                    // THE FIX: Restore pointer events when the mouse is released
                    if (viewerPane) viewerPane.style.pointerEvents = '';

                    document.removeEventListener('mousemove', doDrag);
                    document.removeEventListener('mouseup', stopDrag);
                };

                document.addEventListener('mousemove', doDrag);
                document.addEventListener('mouseup', stopDrag);
            });
        }

        // Initialize Resizers
        initResizable('resizer-folders', 'pane-folders', false);
        initResizable('resizer-files', 'pane-files', false);
        initResizable('resizer-meta', 'pane-meta', true);

        // --- START THE APP ---
        document.addEventListener('DOMContentLoaded', () => {
            loadAppState();
        });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)