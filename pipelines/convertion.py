import os
import glob
import re
import xml.etree.ElementTree as ET
import argparse
import pubmed_parser as pp

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


class ConversionPipeline:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PMC XML to Ground Truth Markdown.")
    parser.add_argument("--data_dir", type=str, default="/home/data1/musong/workspace/2026/03/07/helloworld/data",
                        help="The base directory containing the original PMC case folders with XMLs.")
    parser.add_argument("--out_dir", type=str, default="/home/data1/musong/workspace/2026/03/07/helloworld/log/generated",
                        help="The output directory to save the generated markdown files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = ConversionPipeline(
        data_dir=args.data_dir,
        out_dir=args.out_dir
    )
    
    pipeline.run()