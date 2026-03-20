import os
import requests
import re
from typing import Union, List
import xml.etree.ElementTree as ET

def fetch_ama_citations(dois: Union[str, List[str]], **kwargs) -> str:
    """
    Fetches exact AMA-formatted citation strings for one or multiple DOIs.
    Automatically numbers them sequentially if a list is provided.
    """
    # If a single string is passed, convert it to a list to use the same logic
    if isinstance(dois, str):
        dois = [dois]
        
    headers = {"Accept": "text/x-bibliography; style=american-medical-association"}
    formatted_citations = []
    
    for index, doi in enumerate(dois):
        clean_doi = doi.replace("https://doi.org/", "").replace("doi:", "").strip()
        url = f"https://doi.org/{clean_doi}"
        
        print(f"    [+] Fetching AMA citation for DOI: {clean_doi} ({index + 1}/{len(dois)})")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            raw_citation = response.text.strip()
            
            # Regex to strip leading numbers, dots, and brackets (e.g., "1. ", "[1] ")
            cleaned_citation = re.sub(r'^\[?\d+\]?\.?\s*', '', raw_citation)
            
            # Apply our own sequential numbering
            final_citation = f"{index + 1}. {cleaned_citation}"
            formatted_citations.append(final_citation)
            
            preview = final_citation[:75] + "..." if len(final_citation) > 75 else final_citation
            print(f"        -> Success: {preview}")
            
        except Exception as e:
            error_msg = f"{index + 1}. [Error fetching AMA citation for DOI {clean_doi}: {str(e)}]"
            formatted_citations.append(error_msg)
            print(f"    [X] {error_msg}")
            
    # Join all citations together with double newlines for readability
    return "\n\n".join(formatted_citations)

def search_pubmed(
    query: str, 
    max_results: int = 5, 
    execution_log: dict = None, 
    **kwargs
) -> str:
    """Searches PubMed for recent medical literature using an API key if available."""
    
    exclude_id = execution_log.get("case_id") if execution_log else None
    if exclude_id:
        print(f"    [!] Internal Tool Logic priming to mask Ground Truth (PMID: {exclude_id})")

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    # Retrieve the API key from the OS environment
    api_key = os.environ.get("NCBI_API_KEY")
    
    try:
        search_url = f"{base_url}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results + 2, 
            "sort": "date"  
        }
        
        # Inject API key if it exists
        if api_key:
            search_params["api_key"] = api_key

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
        
        # Inject API key for the fetch request as well
        if api_key:
            fetch_params["api_key"] = api_key

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