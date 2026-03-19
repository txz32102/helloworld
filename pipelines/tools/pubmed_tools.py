import requests
import xml.etree.ElementTree as ET

def fetch_ama_citation(doi: str, **kwargs) -> str:
    """
    Fetches the exact AMA-formatted citation string for a given DOI using standard content negotiation.
    """
    # Clean the DOI in case the LLM passes "https://doi.org/..." or "doi:..."
    clean_doi = doi.replace("https://doi.org/", "").replace("doi:", "").strip()
    url = f"https://doi.org/{clean_doi}"
    
    print(f"    [+] Fetching AMA citation for DOI: {clean_doi}")
    
    # Requesting the American Medical Association format directly from the DOI resolver
    headers = {"Accept": "text/x-bibliography; style=american-medical-association"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # The response is the raw, perfectly formatted AMA citation string
        citation_text = response.text.strip()
        
        # Print a short preview to the console so you know it worked
        preview = citation_text[:75] + "..." if len(citation_text) > 75 else citation_text
        print(f"        -> Success: {preview}")
        
        return citation_text
        
    except Exception as e:
        error_msg = f"Error fetching AMA citation for DOI {clean_doi}: {str(e)}"
        print(f"    [X] {error_msg}")
        return error_msg

def search_pubmed(
    query: str, 
    max_results: int = 5, # Increased default to give the LLM more options 
    execution_log: dict = None, 
    **kwargs
) -> str:
    """Searches PubMed for recent medical literature."""
    
    exclude_id = execution_log.get("case_id") if execution_log else None
    if exclude_id:
        print(f"    [!] Internal Tool Logic priming to mask Ground Truth (PMID: {exclude_id})")

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    try:
        search_url = f"{base_url}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results + 2, 
            "sort": "date"  # CHANGED: Prioritize the newest publications
        }
        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_response.raise_for_status()
        pmids = search_response.json().get("esearchresult", {}).get("idlist", [])
        
        if exclude_id and exclude_id in pmids:
            pmids.remove(exclude_id)
            
        pmids = pmids[:max_results]
        
        if not pmids:
            return f"No results found on PubMed for query: '{query}'"

        fetch_url = f"{base_url}/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml" 
        }
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=10)
        fetch_response.raise_for_status()
        
        root = ET.fromstring(fetch_response.content)
        formatted_results = []
        
        for article in root.findall('.//PubmedArticle'):
            pmid_elem = article.find('.//MedlineCitation/PMID')
            pmid = pmid_elem.text if pmid_elem is not None else "Unknown PMID"
            
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "No Title"
            
            journal_elem = article.find('.//Title')
            journal = journal_elem.text if journal_elem is not None else "Unknown Journal"

            # Added Publication Year extraction
            pub_year_elem = article.find('.//PubDate/Year')
            pub_year = pub_year_elem.text if pub_year_elem is not None else "Unknown Year"
            
            abstract_texts = article.findall('.//AbstractText')
            abstract = " ".join([elem.text for elem in abstract_texts if elem.text])
            if not abstract:
                abstract = "No abstract available."
            
            doi = "No DOI"
            pmcid = "No PMCID"
            for aid in article.findall('.//PubmedData/ArticleIdList/ArticleId'):
                if aid.get('IdType') == 'doi':
                    doi = aid.text
                elif aid.get('IdType') == 'pmc':
                    pmcid = aid.text
                    
            formatted_results.append(
                f"Title: {title}\n"
                f"Journal: {journal} ({pub_year})\n"
                f"PMID: {pmid} | PMCID: {pmcid} | DOI: {doi}\n"
                f"Abstract: {abstract}\n"
            )
                
        return "\n---\n".join(formatted_results)

    except Exception as e:
        return f"    [!] Error: Error querying PubMed database: {str(e)}"
