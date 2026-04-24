import os
import requests
import re
from typing import Union, List
import xml.etree.ElementTree as ET

def _extract_element_text(element) -> str:
    """Safely flattens XML text, including nested tags."""
    if element is None:
        return ""
    return " ".join(part.strip() for part in element.itertext() if part and part.strip()).strip()

def fetch_pubmed_details(
    pmids: Union[str, int, List[Union[str, int]]],
    timeout: int = 10,
    **kwargs
) -> List[dict]:
    """
    Fetches structured PubMed metadata for one or more PMIDs.

    Returned keys:
    - pmid
    - pmcid
    - doi
    - title
    - journal
    - year
    - abstract
    """
    if isinstance(pmids, (str, int)):
        pmids = [pmids]

    clean_pmids = [str(pmid).strip() for pmid in pmids if str(pmid).strip()]
    if not clean_pmids:
        return []

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    api_key = os.environ.get("NCBI_API_KEY")
    fetch_url = f"{base_url}/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(clean_pmids),
        "retmode": "xml"
    }

    if api_key:
        fetch_params["api_key"] = api_key

    response = requests.get(fetch_url, params=fetch_params, timeout=timeout)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    records_by_pmid = {}

    for article in root.findall(".//PubmedArticle"):
        pmid = _extract_element_text(article.find(".//MedlineCitation/PMID"))
        if not pmid:
            continue

        title = _extract_element_text(article.find(".//ArticleTitle"))
        journal = _extract_element_text(article.find(".//Journal/Title"))

        pub_year = _extract_element_text(article.find(".//PubDate/Year"))
        if not pub_year:
            pub_year = _extract_element_text(article.find(".//PubDate/MedlineDate"))
            year_match = re.search(r"\b(19|20)\d{2}\b", pub_year)
            pub_year = year_match.group(0) if year_match else ""

        abstract_parts = []
        for abstract_text in article.findall(".//AbstractText"):
            label = abstract_text.attrib.get("Label")
            text = _extract_element_text(abstract_text)
            if not text:
                continue
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts).strip()

        doi = ""
        pmcid = ""
        for article_id in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
            id_type = article_id.get("IdType")
            value = _extract_element_text(article_id)
            if id_type == "doi" and value:
                doi = value
            elif id_type == "pmc" and value:
                pmcid = value if value.startswith("PMC") else f"PMC{value}"

        records_by_pmid[pmid] = {
            "pmid": pmid,
            "pmcid": pmcid or None,
            "doi": doi or None,
            "title": title or None,
            "journal": journal or None,
            "year": int(pub_year) if str(pub_year).isdigit() else None,
            "abstract": abstract or None,
        }

    return [records_by_pmid[pmid] for pmid in clean_pmids if pmid in records_by_pmid]

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
    """Searches PubMed for recent medical literature, with automatic query relaxation."""
    
    exclude_id = execution_log.get("case_id") if execution_log else None
    if exclude_id:
        print(f"    [!] Internal Tool Logic priming to mask Ground Truth (PMID: {exclude_id})")

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    api_key = os.environ.get("NCBI_API_KEY")
    
    # ---------------------------------------------------------
    # IMPROVEMENT 1: Strip problematic trigger words like "DOI"
    # ---------------------------------------------------------
    clean_query = re.sub(r'\b(doi|pmid)\b', '', query, flags=re.IGNORECASE).strip()
    # Remove extra spaces left behind
    clean_query = re.sub(r'\s+', ' ', clean_query)
    
    # ---------------------------------------------------------
    # IMPROVEMENT 2: The Halving/Relaxation Loop
    # ---------------------------------------------------------
    words = clean_query.split()
    pmids = []
    
    try:
        while words:
            current_query = " ".join(words)
            search_url = f"{base_url}/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": current_query,
                "retmode": "json",
                "retmax": max_results + 2, 
                "sort": "date"  
            }
            
            if api_key:
                search_params["api_key"] = api_key

            search_response = requests.get(search_url, params=search_params, timeout=10)
            search_response.raise_for_status()
            
            pmids = search_response.json().get("esearchresult", {}).get("idlist", [])
            
            if exclude_id and exclude_id in pmids:
                pmids.remove(exclude_id)
                
            pmids = pmids[:max_results]
            
            # If we found results, break out of the loop!
            if pmids:
                # Optional: Print to console so you can see it working
                if len(words) < len(clean_query.split()):
                    print(f"    [+] Query relaxed successfully to: '{current_query}'")
                break
            
            # If no results, halve the list of words for the next attempt
            if len(words) == 1:
                break  # Can't reduce any further
                
            print(f"    [-] No results for '{current_query}'. Halving keywords...")
            # Keep the first half of the words (usually the most important subject terms)
            words = words[:max(1, len(words) // 2)]

        # If it STILL fails after halving all the way down to 1 word
        if not pmids:
            return f"No results found on PubMed for query: '{query}' (even after reducing keywords)."

        # ---------------------------------------------------------
        # THE FETCH LOGIC (Proceeds normally since we now have PMIDs)
        # ---------------------------------------------------------
        fetch_url = f"{base_url}/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml" 
        }
        
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
