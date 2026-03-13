import os
import glob
import re
import json
import xml.etree.ElementTree as ET
import pubmed_parser as pp
import urllib.parse
from jinja2 import Template
from weasyprint import HTML
import markdown

class PMCArticleParser:
    def __init__(self, file_path):
        """Initializes the parser with the path to a PMC XML file."""
        self.file_path = file_path
        self.article_data = {}

    def parse(self):
        """Runs all extraction methods and returns a consolidated dictionary."""
        self._extract_basic_info()
        self._extract_main_content()
        self._extract_figures()
        self._extract_citations()
        return self.article_data

    def _extract_basic_info(self):
        """Extracts title, abstract, and authors."""
        basic_info = pp.parse_pubmed_xml(self.file_path)
        self.article_data['title'] = basic_info.get('full_title', '')
        self.article_data['abstract'] = basic_info.get('abstract', '')
        self.article_data['authors'] = basic_info.get('author_list', [])

    def _extract_main_content(self):
        """Extracts the main text body paragraph by paragraph."""
        paragraphs = pp.parse_pubmed_paragraph(self.file_path)
        self.article_data['main_content'] = [p.get('text', '') for p in paragraphs if p.get('text')]

    def _extract_figures(self):
        """Extracts figure count, labels, captions, and the actual graphic filenames."""
        captions = pp.parse_pubmed_caption(self.file_path)
        
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        
        fig_image_map = {}
        for fig in root.findall('.//fig'):
            fig_id = fig.get('id')
            graphic = fig.find('.//graphic')
            if graphic is not None:
                href = graphic.get('{http://www.w3.org/1999/xlink}href')
                if href:
                    fig_image_map[fig_id] = href

        self.article_data['figure_count'] = len(captions)
        self.article_data['figures'] = []
        
        for fig in captions:
            f_id = fig.get('fig_id', '')
            img_filename = fig_image_map.get(f_id, f_id)
            
            self.article_data['figures'].append({
                'label': fig.get('fig_label', ''),
                'id': f_id,
                'caption': fig.get('fig_caption', ''),
                'img_filename': img_filename  
            })

    def _extract_citations(self):
        """Extracts the reference list and manually parses DOIs from the XML."""
        refs = pp.parse_pubmed_references(self.file_path)
        
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        
        doi_map = {}
        for ref in root.findall('.//ref'):
            ref_id = ref.get('id')
            doi_elem = ref.find('.//pub-id[@pub-id-type="doi"]')
            if doi_elem is not None and doi_elem.text:
                doi_map[ref_id] = doi_elem.text
        
        for ref in refs:
            if not ref.get('doi'):
                ref['doi'] = doi_map.get(ref.get('ref_id'), '')
                
        self.article_data['citations'] = refs


class PMCArticleMDGenerator:
    def __init__(self, article_data, output_path, source_image_dir):
        self.data = article_data
        self.output_path = output_path
        self.source_image_dir = source_image_dir

        # Map images to your new local 'imgs' directory
        for fig in self.data.get('figures', []):
            base_img = fig['img_filename']
            # Look in the source dir just to find the actual file with its extension
            matched_files = glob.glob(os.path.join(self.source_image_dir, f"{base_img}*"))
            
            valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff')
            valid_images = [f for f in matched_files if f.lower().endswith(valid_extensions)]
            
            if valid_images:
                # Grab just the filename (e.g., "10014_2020_365_Fig1_HTML.jpg")
                actual_filename = os.path.basename(valid_images[0])
                # Path is now just relative to the current folder where the MD file lives
                fig['img_path'] = f"imgs/{actual_filename}"

    def _get_interleaved_content(self):
        """Mixes paragraphs and figures logically, ensuring sequential order."""
        blocks = []
        placed_figures = set()

        def get_fig_num(fig):
            match = re.search(r'\d+', fig.get('label', '') or fig.get('id', ''))
            return int(match.group()) if match else 999
            
        sorted_figures = sorted(self.data.get('figures', []), key=get_fig_num)

        for para in self.data.get('main_content', []):
            blocks.append({'type': 'paragraph', 'content': para})
            
            for fig in sorted_figures:
                if fig['id'] in placed_figures:
                    continue
                
                label_text = fig.get('label', '') or fig.get('id', '')
                match = re.search(r'\d+', label_text)
                if match:
                    fig_num = match.group()
                    pattern = r'\b(?:fig\.?|figure|figs\.?|figures)\b[^.]{0,40}?\b' + fig_num + r'[a-zA-Z]*\b'
                    
                    if re.search(pattern, para, re.IGNORECASE):
                        blocks.append({'type': 'figure', 'content': fig})
                        placed_figures.add(fig['id'])

        # Catch-all for remaining figures
        for fig in sorted_figures:
            if fig['id'] not in placed_figures:
                blocks.append({'type': 'figure', 'content': fig})

        return blocks

    def generate(self):
        """Generates the Markdown file and writes it to disk."""
        interleaved_blocks = self._get_interleaved_content()
        md_lines = []

        # Title
        title = self.data.get('title', 'Untitled Document')
        md_lines.append(f"# {title}\n")

        # Authors
        authors = self.data.get('authors', [])
        author_names = []
        for a in authors:
            if isinstance(a, list) and len(a) >= 2:
                last_name = a[0] if a[0] else ""
                first_name = a[1] if a[1] else ""
                name = f"{first_name} {last_name}".strip()
                if name:
                    author_names.append(name)
            elif isinstance(a, dict):
                last_name = a.get('last_name', '') or ""
                first_name = a.get('first_name', '') or ""
                name = f"{first_name} {last_name}".strip()
                if name:
                    author_names.append(name)

        # Deduplicate authors while preserving order
        unique_authors = []
        seen = set()
        for name in author_names:
            if name not in seen:
                unique_authors.append(name)
                seen.add(name)

        if unique_authors:
            md_lines.append(f"**Authors:** {', '.join(unique_authors)}\n")

        # Abstract
        abstract = self.data.get('abstract', '')
        if abstract:
            md_lines.append("## Abstract\n")
            md_lines.append(f"{abstract}\n")

        # Main Content
        md_lines.append("## Main Content\n")
        for block in interleaved_blocks:
            if block['type'] == 'paragraph':
                md_lines.append(f"{block['content']}\n")
            elif block['type'] == 'figure':
                fig = block['content']
                img_path = fig.get('img_path', '')
                label = fig.get('label', 'Figure')
                caption = fig.get('caption', '')
                
                # Render Image if path exists
                if img_path:
                    md_lines.append(f"![{label}]({img_path})\n")
                
                # Render Caption
                md_lines.append(f"> **{label}:** {caption}\n")

        # References
        citations = self.data.get('citations', [])
        if citations:
            md_lines.append("## References\n")
            for idx, cite in enumerate(citations, 1):
                authors_cite = cite.get('authors', '')
                year = cite.get('year', '')
                title = cite.get('article_title', '')
                journal = cite.get('journal', '')
                doi = cite.get('doi', '')

                ref_str = f"{idx}. {authors_cite} ({year}). \"{title}\". *{journal}*."
                if doi:
                    ref_str += f" [DOI: {doi}](https://doi.org/{doi})"
                
                md_lines.append(f"{ref_str}\n")

        # Write out to file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
            
        print(f"✅ Successfully saved Ground Truth Markdown to: {self.output_path}\n")


class MDConversionPipeline:
    def __init__(self, data_dir: str, out_dir: str):
        """
        Initializes the pipeline to convert PMC XML to Ground Truth Markdown.
        """
        self.data_dir = data_dir
        self.out_dir = out_dir

    def run(self):
        """
        Executes the batch processing loop.
        """
        print(f"Looking for XML files in: {self.data_dir}")
        xml_files = glob.glob(os.path.join(self.data_dir, "*", "*.xml"))
        
        if not xml_files:
            print("[-] No XML files found!")
            return
            
        print(f"Found {len(xml_files)} XML file(s). Starting conversion...\n")
        print("-" * 40)
        
        for file_path in xml_files:
            doc_id = os.path.basename(os.path.dirname(file_path))
            
            target_doc_dir = os.path.join(self.out_dir, doc_id)
            os.makedirs(target_doc_dir, exist_ok=True) 
            
            output_md_path = os.path.join(target_doc_dir, f"{doc_id}_gt.md")
            source_image_dir = os.path.dirname(file_path) 
                
            print(f"Processing document: {doc_id}...")
            
            try:
                parser = PMCArticleParser(file_path)
                article_data = parser.parse()
                
                generator = PMCArticleMDGenerator(article_data, output_md_path, source_image_dir)
                generator.generate()
            except Exception as e:
                print(f"[X] Failed to process document {doc_id}: {e}\n")
                
        print("-" * 40)
        print("Batch ground-truth conversion complete.")
        
class PDFConversionPipeline:
    def __init__(self, working_dir: str, fonts_dir: str):
        """
        Initializes the pipeline to generate PDFs from Markdown and JSON assets.
        
        :param working_dir: Path to the log/pipelines directory containing the article folders.
        :param fonts_dir: Path to the local fonts directory.
        """
        self.working_dir = working_dir
        self.fonts_dir = fonts_dir
        
        self.font_regular_uri = self._get_font_uri("Times New Roman.ttf")
        self.font_bold_uri = self._get_font_uri("Times New Roman - Bold.ttf")
        self.template = Template(self._get_html_template())

    def _get_font_uri(self, font_name: str) -> str:
        """Safely encodes the local font path for CSS consumption."""
        font_path = os.path.join(self.fonts_dir, font_name)
        return f"file://{urllib.parse.quote(font_path)}"

    def _get_html_template(self) -> str:
        """Returns the base Jinja2 HTML template."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @font-face {
                    font-family: 'LocalTimesNewRoman';
                    src: url('{{ font_regular_uri }}') format('truetype');
                    font-weight: normal;
                    font-style: normal;
                }
                @font-face {
                    font-family: 'LocalTimesNewRoman';
                    src: url('{{ font_bold_uri }}') format('truetype');
                    font-weight: bold;
                    font-style: normal;
                }

                @page { size: A4; margin: 2cm; }
                
                body { 
                    font-family: 'LocalTimesNewRoman', serif; 
                    line-height: 1.6; 
                    color: #333; 
                    font-size: 12pt; 
                }
                
                h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 5px; }
                h2, h3 { color: #2c3e50; margin-top: 20px; }
                
                pre { 
                    background-color: #f8f9fa; 
                    padding: 15px; 
                    border-radius: 5px; 
                    font-size: 10pt; 
                    border: 1px solid #e9ecef; 
                    white-space: pre-wrap;       
                    word-wrap: break-word;       
                    overflow-wrap: break-word;   
                }
                code { font-family: Consolas, monospace; }
                
                .supplementary-section { page-break-before: always; }
                
                .figure-container {
                    page-break-inside: avoid; /* Keeps image and caption together */
                    margin: 25px 0;
                    width: 100%; 
                    text-align: center;
                }
                
                .figure-container img {
                    /* Natural scaling: Shrinks if it's too wide, but maintains aspect ratio */
                    max-width: 100%; 
                    height: auto; 
                    
                    /* A fail-safe max-height so extremely tall images don't break the page, 
                       but standard images won't be forced to this size */
                    max-height: 12cm; 
                    
                    /* Centering */
                    display: block;
                    margin: 0 auto 10px auto;
                }
                
                .figure-container .caption {
                    text-align: left;
                    font-size: 11pt;
                    color: #444;
                    line-height: 1.4;
                    margin-top: 8px;
                    page-break-inside: avoid;
                }
            </style>
        </head>
        <body>
            {{ body_html }}
        </body>
        </html>
        """
        
    def run(self):
        """Iterates over the generated pipelines directory and processes each article."""
        if not os.path.exists(self.working_dir):
            print(f"[-] Directory not found: {self.working_dir}")
            return

        print(f"Looking for article directories in: {self.working_dir}")
        for item in os.listdir(self.working_dir):
            article_dir = os.path.join(self.working_dir, item)
            
            if os.path.isdir(article_dir):
                self._process_article(item, article_dir)
                
        print("-" * 40)
        print("Batch PDF generation complete.")

    def _process_article(self, article_id: str, article_dir: str):
        """Processes an individual article into a PDF."""
        print(f"Processing PDF for Article ID: {article_id}...")
        
        # 1. Define specific file paths based on new tree structure
        generated_md_path = os.path.join(article_dir, f"{article_id}_generated.md")
        eval_md_path = os.path.join(article_dir, f"{article_id}_evaluation.md")
        json_path = os.path.join(article_dir, f"{article_id}_atoms.json")
        pdf_out_path = os.path.join(article_dir, f"{article_id}_report.pdf")

        # 2. Check for the core generated markdown file
        if not os.path.exists(generated_md_path):
            print(f"  ⚠️ Skipping {article_id}: {generated_md_path} not found.")
            return

        # 3. Read Core Assets
        with open(generated_md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # 4. Process Images in the main content
        md_content = self._process_images(md_content, article_dir)

        # 5. Build the Supplementary Section
        md_content += "\n\n<div class='supplementary-section'></div>\n\n"
        md_content += "---\n\n# Supplementary Material\n\n"

        # Append Evaluation MD if it exists
        if os.path.exists(eval_md_path):
            with open(eval_md_path, 'r', encoding='utf-8') as f:
                eval_content = f.read()
            md_content += eval_content + "\n\n"
        else:
            print(f"  ℹ️ Notice: No evaluation file found for {article_id}.")

        # Append JSON metadata if it exists
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            md_content += "### Execution Atoms (JSON)\n\n"
            md_content += "```json\n" + json.dumps(json_data, indent=2) + "\n```\n"
        else:
            print(f"  ℹ️ Notice: No atoms JSON file found for {article_id}.")

        # 6. Convert to HTML and Render Jinja
        raw_html = markdown.markdown(md_content, extensions=['fenced_code', 'tables'])
        final_html = self.template.render(
            body_html=raw_html,
            font_regular_uri=self.font_regular_uri,
            font_bold_uri=self.font_bold_uri
        )

        # 7. Generate PDF
        try:
            HTML(string=final_html, base_url=article_dir).write_pdf(pdf_out_path)
            print(f"  ✅ Success! Saved to: {pdf_out_path}\n")
        except Exception as e:
            print(f"  ❌ Failed to generate PDF for {article_id}. Error: {e}\n")

    def _process_images(self, md_content: str, article_dir: str) -> str:
        """Parses explicitly defined Markdown images and their captions to build HTML figures."""
        
        def replace_with_figure(match):
            alt_text = match.group(1)      # e.g., Fig. 1
            img_rel_path = match.group(2)  # e.g., imgs/10014_2020_365_Fig1_HTML.jpg
            caption_prefix = match.group(3) # e.g., Fig. 1:
            caption_text = match.group(4)   # e.g., Radiological findings...
            
            # Clean up trailing colons in the prefix if they got caught inside the bold tags
            caption_prefix = caption_prefix.strip(" :")
            
            # Extract just the filename and build the absolute local path
            img_basename = os.path.basename(img_rel_path)
            img_path = os.path.abspath(os.path.join(article_dir, "imgs", img_basename))
            
            # Safely encode the URI for WeasyPrint
            img_uri = f"file://{urllib.parse.quote(img_path)}"
            
            # Construct the custom HTML block
            return (
                f'<div class="figure-container">\n'
                f'  <img src="{img_uri}" alt="{alt_text}" />\n'
                f'  <div class="caption"><strong>{caption_prefix}:</strong> {caption_text}</div>\n'
                f'</div>\n'
            )

        # Robust Regex Pattern Explanation:
        # 1. !\[([^\]]*)\]\(([^)]+)\)  -> Matches ![alt_text](path/to/img)
        # 2. \s* -> Matches ANY whitespace or newlines between image and caption
        # 3. (?:>\s*)?                -> Optionally matches a blockquote marker "> "
        # 4. \*\*([^*]+)\*\* -> Matches **Caption Prefix**
        # 5. \s*:?\s* -> Matches optional colons/spaces after the bold text
        # 6. (.*)                     -> Matches the rest of the caption text
        
        pattern = r'!\[([^\]]*)\]\(([^)]+)\)\s*(?:>\s*)?\*\*([^*]+)\*\*\s*:?\s*(.*)'
        
        return re.sub(pattern, replace_with_figure, md_content)