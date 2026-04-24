import heapq
import json
import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import requests
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - keeps retrieval-only mode usable without the SDK.
    OpenAI = None

from .pubmed_tools import fetch_pubmed_details

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISEASE_INDEX_PATH = PROJECT_ROOT / "log" / "preprocessing" / "extracted_diseases.jsonl"
DEFAULT_LLM_FILTERED_DB_PATH = PROJECT_ROOT / "data" / "llm_filtered.db"
DEFAULT_SOURCE_METADATA_DB_PATH = PROJECT_ROOT / "data" / "source_metadata.db"
DEFAULT_TITLE_ABSTRACT_DB_DIR = PROJECT_ROOT / "data" / "title_abstract_db"
PMC_OA_BASE_URL = "https://pmc-oa-opendata.s3.amazonaws.com"

STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of", "on",
    "or", "the", "to", "with", "without"
}
GENERIC_QUERY_PHRASES = {
    "rare",
    "benign",
    "malignant",
    "benign tumor",
    "malignant tumor",
    "tumor",
    "tumour",
    "mass",
    "lesion",
    "case",
    "case report",
    "report",
}
GENERIC_SINGLE_TOKENS = {
    "rare", "benign", "malignant", "tumor", "tumour", "mass", "lesion", "cancer",
    "disease", "syndrome", "report", "case", "patient", "presentation", "adult",
    "male", "female", "child", "liver", "lung", "brain", "kidney", "renal",
    "hepatic", "spinal", "spine", "bone", "skin", "heart", "cardiac", "gastric",
    "intestinal", "colon", "rectal", "biliary", "thyroid", "breast", "ovarian",
    "uterine", "prostate", "pancreatic",
}
HIGH_IMPORTANCE_HINTS = (
    "misdiagn", "pregnan", "pediatric", "paediatric", "fatal", "life-threatening",
    "hemodynamic", "haemodynamic", "refractory", "complication", "diagnostic challenge",
    "delayed diagnosis", "severe", "critical", "icu",
)


def _normalize_text(text: Any) -> str:
    clean = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", clean).strip()


def _tokenize(text: Any) -> List[str]:
    return [
        token for token in _normalize_text(text).split()
        if len(token) > 1 and token not in STOPWORDS
    ]


def _truncate(text: Any, limit: int = 800) -> Optional[str]:
    value = str(text or "").strip()
    if not value:
        return None
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _dedupe_preserve_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    ordered = []
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = re.split(r"[\n,;]+", value)
    elif isinstance(value, (list, tuple, set)):
        raw = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                raw.extend(re.split(r"[\n,;]+", item))
            else:
                raw.append(str(item))
    else:
        raw = [str(value)]
    return [item.strip() for item in raw if str(item).strip()]


def _resolve_repo_path(path_value: Any) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(str(path_value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _is_generic_phrase(phrase: str) -> bool:
    normalized = _normalize_text(phrase)
    if not normalized:
        return True
    if normalized in GENERIC_QUERY_PHRASES:
        return True
    tokens = normalized.split()
    if len(tokens) == 1 and tokens[0] in GENERIC_SINGLE_TOKENS:
        return True
    return False


def _clean_query_phrases(phrases: Iterable[str]) -> List[str]:
    normalized = []
    for phrase in phrases:
        clean = _normalize_text(phrase)
        if clean:
            normalized.append(clean)
    unique = _dedupe_preserve_order(normalized)
    specific = [phrase for phrase in unique if not _is_generic_phrase(phrase)]
    return specific or unique


def _is_specific_partial_match(left: str, right: str) -> bool:
    if not left or not right or left == right:
        return False
    if left not in right and right not in left:
        return False

    shorter = left if len(left) <= len(right) else right
    tokens = shorter.split()

    if len(tokens) >= 2:
        return True
    if len(shorter) >= 8 and shorter not in GENERIC_SINGLE_TOKENS:
        return True
    return False


def _citation_status(record: Dict[str, Any]) -> str:
    if record.get("doi"):
        return "doi_ready"
    if record.get("pmid") or record.get("pmcid"):
        return "traceable_without_doi"
    return "metadata_incomplete"


def _paper_id(record: Dict[str, Any]) -> str:
    if record.get("pmcid"):
        return str(record["pmcid"])
    if record.get("pmc_id"):
        return f"PMC{record['pmc_id']}"
    if record.get("pmid"):
        return f"PMID{record['pmid']}"
    return f"record_{abs(hash(json.dumps(record, sort_keys=True, default=str))) % 1_000_000}"


@lru_cache(maxsize=1)
def _load_disease_index(index_path: str = str(DEFAULT_DISEASE_INDEX_PATH)) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    path = Path(index_path)
    records = []
    by_pmc_id = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pmc_id_raw = row.get("pmc_id")
            try:
                pmc_id = int(str(pmc_id_raw).replace("PMC", ""))
            except Exception:
                continue

            diseases = [str(item).strip() for item in row.get("diseases", []) if str(item).strip()]
            normalized_phrases = _clean_query_phrases(diseases)
            token_set = set()
            for phrase in normalized_phrases:
                token_set.update(_tokenize(phrase))

            record = {
                "pmc_id": pmc_id,
                "year": row.get("year"),
                "diseases": diseases,
                "normalized_phrases": normalized_phrases,
                "token_set": token_set,
            }
            records.append(record)
            by_pmc_id[pmc_id] = record

    return records, by_pmc_id


def _safe_load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_case_source_metadata(case_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(case_data, dict):
        return {}

    source_dir = _resolve_repo_path(case_data.get("metadata", {}).get("source_directory"))
    if not source_dir:
        return {}

    metadata_path = source_dir / "metadata.json"
    metadata = _safe_load_json(metadata_path)
    if metadata:
        metadata["source_directory"] = str(source_dir)
    return metadata


def _format_case_data_context(case_data: Optional[Dict[str, Any]], source_metadata: Dict[str, Any]) -> str:
    parts = []

    title = source_metadata.get("title")
    abstract = source_metadata.get("abstract")
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")

    if isinstance(case_data, dict):
        for section_name in ("history", "presentation", "diagnostics", "management", "outcome", "diagnosis"):
            value = case_data.get(section_name)
            if isinstance(value, list):
                clean_items = [str(item).strip() for item in value if str(item).strip()]
                if clean_items:
                    parts.append(f"{section_name.capitalize()}: " + " ; ".join(clean_items[:6]))
            elif isinstance(value, dict):
                pairs = [f"{k}: {v}" for k, v in value.items() if str(v).strip()]
                if pairs:
                    parts.append(f"{section_name.capitalize()}: " + " ; ".join(pairs[:6]))
            elif isinstance(value, str) and value.strip():
                parts.append(f"{section_name.capitalize()}: {value.strip()}")

    return _truncate("\n".join(parts), 3000) or ""


def _build_query_bundle(
    diseases: Any,
    related_keywords: Any,
    case_context: Optional[str],
    case_data: Optional[Dict[str, Any]],
    current_pmc_id: Optional[Union[int, str]] = None,
    current_pmid: Optional[Union[int, str]] = None,
) -> Dict[str, Any]:
    explicit_diseases = _coerce_string_list(diseases)
    explicit_keywords = _coerce_string_list(related_keywords)
    source_metadata = _load_case_source_metadata(case_data)

    inferred_pmc_id = current_pmc_id or source_metadata.get("pmc_id")
    inferred_pmid = current_pmid or source_metadata.get("pmid")

    try:
        inferred_pmc_id = int(str(inferred_pmc_id).replace("PMC", "")) if inferred_pmc_id else None
    except Exception:
        inferred_pmc_id = None

    _, by_pmc_id = _load_disease_index()
    index_row = by_pmc_id.get(inferred_pmc_id) if inferred_pmc_id else None
    indexed_case_diseases = list(index_row.get("diseases", [])) if index_row else []

    diagnosis_terms = []
    if isinstance(case_data, dict):
        diagnosis_terms = _coerce_string_list(case_data.get("diagnosis"))

    query_diseases = _dedupe_preserve_order(explicit_diseases + indexed_case_diseases + diagnosis_terms)
    query_phrases = _clean_query_phrases(query_diseases)

    keyword_phrases = _clean_query_phrases(explicit_keywords)

    final_case_context = str(case_context or "").strip()
    if not final_case_context:
        final_case_context = _format_case_data_context(case_data, source_metadata)

    return {
        "current_pmc_id": inferred_pmc_id,
        "current_pmid": inferred_pmid,
        "query_diseases": query_diseases,
        "query_phrases": query_phrases,
        "related_keywords": explicit_keywords,
        "keyword_phrases": keyword_phrases,
        "case_context": final_case_context,
        "source_metadata": source_metadata,
    }


def _score_retrieval_candidate(
    query_phrases: List[str],
    keyword_phrases: List[str],
    query_tokens: set,
    candidate: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    candidate_phrases = candidate.get("normalized_phrases", [])
    candidate_token_set = candidate.get("token_set", set())

    exact_matches = sorted(set(query_phrases) & set(candidate_phrases))
    partial_matches = []
    for query_phrase in query_phrases:
        if query_phrase in exact_matches:
            continue
        for candidate_phrase in candidate_phrases:
            if _is_specific_partial_match(query_phrase, candidate_phrase):
                partial_matches.append({
                    "query_phrase": query_phrase,
                    "matched_phrase": candidate_phrase,
                })
                break

    keyword_matches = []
    for keyword_phrase in keyword_phrases:
        for candidate_phrase in candidate_phrases:
            if keyword_phrase == candidate_phrase:
                keyword_matches.append(candidate_phrase)
                break
            if _is_specific_partial_match(keyword_phrase, candidate_phrase):
                keyword_matches.append(candidate_phrase)
                break

    overlapping_tokens = sorted(query_tokens & candidate_token_set)
    if not exact_matches and not partial_matches and not keyword_matches and not overlapping_tokens:
        return None

    score = (
        len(exact_matches) * 12.0
        + len(partial_matches) * 5.0
        + len(set(keyword_matches)) * 3.0
        + len(overlapping_tokens) * 0.75
    )
    score += min(sum(len(item["matched_phrase"].split()) for item in partial_matches) * 0.2, 2.0)

    return {
        "score": round(score, 4),
        "match_details": {
            "exact_phrase_matches": exact_matches,
            "partial_phrase_matches": partial_matches,
            "keyword_phrase_matches": sorted(set(keyword_matches)),
            "token_overlap": overlapping_tokens,
        },
    }


def _select_rows_by_pmc_ids(db_path: Path, pmc_ids: List[int], columns: List[str]) -> Dict[int, Dict[str, Any]]:
    if not pmc_ids or not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in pmc_ids)
        query = f"SELECT {', '.join(columns)} FROM publications WHERE pmc_id IN ({placeholders})"
        rows = conn.execute(query, pmc_ids).fetchall()
        return {
            int(row["pmc_id"]): dict(row)
            for row in rows
            if row["pmc_id"] is not None
        }
    finally:
        conn.close()


def _fetch_title_abstract_row(pmc_id: int, year: Any) -> Dict[str, Any]:
    try:
        year_value = int(year)
    except Exception:
        return {}

    db_path = DEFAULT_TITLE_ABSTRACT_DB_DIR / f"pub_abstracts_{year_value}.db"
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT pmc_id, title, abstract FROM publications WHERE pmc_id = ? LIMIT 1",
            (pmc_id,),
        ).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def _fetch_local_metadata(pmc_ids: List[int], disease_rows: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    llm_columns = [
        "pmc_id", "pmid", "year", "journal", "pub_date", "volume", "pages", "license",
        "title", "abstract", "category", "is_case_report", "rarity_level", "reasoning"
    ]
    source_columns = [
        "pmc_id", "pmid", "year", "journal", "pub_date", "volume", "pages", "license", "file_path"
    ]

    llm_rows = _select_rows_by_pmc_ids(DEFAULT_LLM_FILTERED_DB_PATH, pmc_ids, llm_columns)
    source_rows = _select_rows_by_pmc_ids(DEFAULT_SOURCE_METADATA_DB_PATH, pmc_ids, source_columns)

    merged = {}
    for pmc_id in pmc_ids:
        combined = {}
        if pmc_id in source_rows:
            combined.update(source_rows[pmc_id])

        title_abstract_row = {}
        if pmc_id not in llm_rows:
            year = combined.get("year") or disease_rows.get(pmc_id, {}).get("year")
            title_abstract_row = _fetch_title_abstract_row(pmc_id, year)
            if title_abstract_row:
                combined.update(title_abstract_row)

        if pmc_id in llm_rows:
            combined.update(llm_rows[pmc_id])

        if combined:
            combined["pmc_id"] = pmc_id
            combined["pmcid"] = f"PMC{pmc_id}"
            merged[pmc_id] = combined

    return merged


def _fetch_pubmed_backfill(pmids: List[Union[str, int]], warnings: List[str]) -> Dict[str, Dict[str, Any]]:
    clean_pmids = [str(pmid).strip() for pmid in pmids if str(pmid).strip()]
    if not clean_pmids:
        return {}
    try:
        return {
            str(record["pmid"]): record
            for record in fetch_pubmed_details(clean_pmids)
            if record.get("pmid")
        }
    except Exception as exc:
        warnings.append(f"PubMed DOI backfill failed: {exc}")
        return {}


def _fetch_full_text_snippet(pmc_id: Union[int, str], max_chars: int = 5000, timeout: int = 20) -> Optional[str]:
    pmc_numeric = str(pmc_id).replace("PMC", "").strip()
    if not pmc_numeric:
        return None

    metadata_url = f"{PMC_OA_BASE_URL}/metadata/PMC{pmc_numeric}.1.json"
    metadata_response = requests.get(metadata_url, timeout=timeout)
    if metadata_response.status_code != 200:
        return None

    xml_url = metadata_response.json().get("xml_url")
    if not xml_url:
        return None

    clean_path = xml_url.replace("s3://pmc-oa-opendata/", "").split("?")[0]
    download_url = f"{PMC_OA_BASE_URL}/{clean_path}"
    xml_response = requests.get(download_url, timeout=timeout)
    xml_response.raise_for_status()

    root = ET.fromstring(xml_response.content)
    parts = []
    for xpath in (".//abstract", ".//body"):
        node = root.find(xpath)
        if node is None:
            continue
        text = " ".join(chunk.strip() for chunk in node.itertext() if chunk and chunk.strip())
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        remaining = max_chars - sum(len(part) for part in parts)
        if remaining <= 0:
            break
        parts.append(text[:remaining])

    full_text = " ".join(parts).strip()
    return full_text[:max_chars] if full_text else None


def _project_retrieved_record(record: Dict[str, Any]) -> Dict[str, Any]:
    pmcid = record.get("pmcid")
    if not pmcid and record.get("pmc_id"):
        pmcid = f"PMC{record['pmc_id']}"

    projected = {
        "paper_id": _paper_id(record),
        "pmc_id": record.get("pmc_id"),
        "pmid": record.get("pmid"),
        "pmcid": pmcid,
        "doi": record.get("doi"),
        "citation_status": _citation_status(record),
        "year": record.get("year"),
        "journal": record.get("journal"),
        "pub_date": record.get("pub_date"),
        "title": record.get("title"),
        "abstract_excerpt": _truncate(record.get("abstract"), 700),
        "diseases": record.get("diseases", []),
        "match_score": record.get("score"),
        "match_details": record.get("match_details", {}),
        "rarity_level": record.get("rarity_level"),
        "category": record.get("category"),
        "local_rarity_reasoning": _truncate(record.get("local_rarity_reasoning"), 260),
    }

    if record.get("full_text_excerpt"):
        projected["full_text_excerpt"] = _truncate(record.get("full_text_excerpt"), 700)

    return projected


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _sanitize_level(value: Any, default: str = "medium") -> str:
    normalized = _normalize_text(value)
    return normalized if normalized in {"high", "medium", "low"} else default


def _coerce_claim_list(value: Any) -> List[str]:
    if isinstance(value, list):
        claims = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str):
        claims = [item.strip("- ").strip() for item in re.split(r"[\n;]+", value) if item.strip()]
    else:
        claims = []
    return _dedupe_preserve_order(claims)


def _build_openai_client(llm_base_url: Optional[str], api_key_env: str) -> Optional[Any]:
    if OpenAI is None:
        return None
    resolved_base_url = llm_base_url or os.environ.get("OPENAI_BASE_URL")
    actual_api_key = os.environ.get(api_key_env)
    if not actual_api_key:
        actual_api_key = api_key_env if api_key_env and api_key_env != "OPENAI_API_KEY" else None

    if resolved_base_url:
        return OpenAI(api_key=actual_api_key or "EMPTY", base_url=resolved_base_url)

    if actual_api_key:
        return OpenAI(api_key=actual_api_key)

    return None


def _run_llm_evidence_filter(
    query_bundle: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    llm_model: str,
    llm_base_url: Optional[str],
    api_key_env: str,
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    client = _build_openai_client(llm_base_url=llm_base_url, api_key_env=api_key_env)
    if client is None:
        warnings.append(
            "LLM filtering skipped because no OpenAI-compatible credentials or base URL were configured."
        )
        return None

    current_case_payload = {
        "pmc_id": query_bundle.get("current_pmc_id"),
        "pmid": query_bundle.get("current_pmid"),
        "query_diseases": query_bundle.get("query_diseases", []),
        "related_keywords": query_bundle.get("related_keywords", []),
        "title": query_bundle.get("source_metadata", {}).get("title"),
        "abstract_excerpt": _truncate(query_bundle.get("source_metadata", {}).get("abstract"), 1200),
        "case_context_excerpt": _truncate(query_bundle.get("case_context"), 1500),
    }
    candidate_payload = []
    for candidate in candidates:
        candidate_payload.append({
            "paper_id": candidate["paper_id"],
            "year": candidate.get("year"),
            "title": candidate.get("title"),
            "abstract_excerpt": _truncate(candidate.get("abstract_excerpt"), 1200),
            "full_text_excerpt": _truncate(candidate.get("full_text_excerpt"), 1800),
            "diseases": candidate.get("diseases", []),
            "retrieval_score": candidate.get("match_score"),
            "match_details": candidate.get("match_details", {}),
            "doi": candidate.get("doi"),
            "pmid": candidate.get("pmid"),
            "pmcid": candidate.get("pmcid"),
            "rarity_level": candidate.get("rarity_level"),
            "category": candidate.get("category"),
        })

    prompt = f"""You are a conservative medical-literature triage assistant for case-report writing.

Your job is to judge how much the retrieved prior papers support claims about rarity, novelty, and clinical importance for the CURRENT CASE.

Be strict:
- Prefer conservative wording.
- Do not support a "first reported case" claim unless the evidence clearly justifies it.
- If close prior cases exist, recommend wording like "rare", "few prior reports", or "unusual presentation" instead.
- Keep citationability intact. If you keep a paper, you must reference it by the exact `paper_id` provided.
- Use only the candidate papers below. Do not invent papers.

Return JSON only with this schema:
{{
  "summary": "1 short paragraph",
  "novelty_level": "high|medium|low",
  "importance_level": "high|medium|low",
  "supported_claim_candidates": ["claim 1", "claim 2"],
  "caveats": ["caveat 1", "caveat 2"],
  "uncertainty": "1 short sentence",
  "selected_papers": [
    {{
      "paper_id": "exact id from the candidate list",
      "relevance": "high|medium|low",
      "supports": ["rare disease", "unusual presentation", "diagnostic challenge"],
      "reason": "why this paper matters"
    }}
  ]
}}

CURRENT CASE:
{json.dumps(current_case_payload, ensure_ascii=False, indent=2)}

CANDIDATE PRIOR PAPERS:
{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}
"""

    request_kwargs = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if not llm_model.startswith("gpt-5"):
        request_kwargs["temperature"] = 0.1

    try:
        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content if response.choices else ""
        parsed = _extract_json_object(content)
        if not parsed:
            warnings.append("LLM filtering returned non-JSON output, so heuristic fallback was used.")
        return parsed
    except Exception as exc:
        warnings.append(f"LLM filtering failed: {exc}")
        return None


def _fallback_assessment(query_bundle: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    top_candidates = candidates[:4]
    exact_hits = sum(len(item.get("match_details", {}).get("exact_phrase_matches", [])) for item in top_candidates)
    partial_hits = sum(len(item.get("match_details", {}).get("partial_phrase_matches", [])) for item in top_candidates)
    rare_hits = sum(
        1 for item in top_candidates
        if item.get("rarity_level") in {"rare_disease", "unprecedented_presentation"}
    )
    context = _normalize_text(query_bundle.get("case_context"))

    if exact_hits >= 3:
        novelty_level = "low"
        summary = "Multiple close precedent papers were retrieved, so this case looks more like a known pattern than a truly novel report."
        claim_candidates = [
            "Similar cases have been reported; frame this as an instructive or diagnostically challenging case rather than a first case."
        ]
    elif exact_hits >= 1 or partial_hits >= 3:
        novelty_level = "medium"
        summary = "The retrieval found partly overlapping prior cases, which may support an 'unusual presentation' or 'few prior reports' framing."
        claim_candidates = [
            "This may support wording such as 'rare' or 'unusual presentation', but not a definitive 'first reported case' claim."
        ]
    else:
        novelty_level = "medium"
        summary = "The local disease index did not surface many close precedent cases, but that absence alone is not enough to claim true novelty."
        claim_candidates = [
            "Few close matches were retrieved from the local disease index; broader literature verification is still required before making rarity claims."
        ]

    importance_level = "medium"
    if rare_hits >= 2 or any(hint in context for hint in HIGH_IMPORTANCE_HINTS):
        importance_level = "high"
    elif exact_hits >= 3 and rare_hits == 0:
        importance_level = "low"

    caveats = [
        "This fallback assessment used retrieval heuristics instead of an LLM evidence filter.",
        "The local disease index is a first-pass screen and is not sufficient to support a 'first reported case' claim by itself.",
    ]
    if not top_candidates:
        caveats.append("No similar cases were retrieved, so the output is especially uncertain.")

    return {
        "summary": summary,
        "novelty_level": novelty_level,
        "importance_level": importance_level,
        "supported_claim_candidates": claim_candidates,
        "caveats": caveats,
        "uncertainty": "Use broader literature review before making any strong novelty claim.",
    }


def assess_disease_importance(
    diseases: Any = None,
    related_keywords: Any = None,
    case_context: Optional[str] = None,
    top_k: int = 8,
    fetch_full_text: bool = False,
    run_llm_filter: bool = True,
    llm_model: str = "gpt-4.1",
    llm_base_url: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    case_data: Optional[Dict[str, Any]] = None,
    execution_log: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Retrieves similar disease-focused prior cases, preserves citation metadata,
    and optionally uses an LLM to judge how strong a novelty/importance claim is.
    """
    warnings = []
    query_bundle = _build_query_bundle(
        diseases=diseases,
        related_keywords=related_keywords,
        case_context=case_context,
        case_data=case_data,
    )

    query_phrases = query_bundle["query_phrases"]
    keyword_phrases = query_bundle["keyword_phrases"]
    query_tokens = set()
    for phrase in query_phrases + keyword_phrases:
        query_tokens.update(_tokenize(phrase))

    if not query_phrases and not keyword_phrases:
        return {
            "error": (
                "No disease query could be inferred. Provide `diseases` explicitly or pass "
                "case_data with a source_directory/diagnosis field."
            ),
            "warnings": warnings,
        }

    try:
        top_k = max(1, min(int(top_k), 20))
    except Exception:
        top_k = 8

    index_records, _ = _load_disease_index()
    current_pmc_id = query_bundle.get("current_pmc_id")

    ranked_heap = []
    disease_rows = {}
    for record in index_records:
        if current_pmc_id and record["pmc_id"] == current_pmc_id:
            continue

        scoring = _score_retrieval_candidate(
            query_phrases=query_phrases,
            keyword_phrases=keyword_phrases,
            query_tokens=query_tokens,
            candidate=record,
        )
        if not scoring:
            continue

        enriched = {
            **record,
            **scoring,
        }
        disease_rows[record["pmc_id"]] = enriched
        entry = (scoring["score"], record.get("year") or 0, record["pmc_id"])
        if len(ranked_heap) < top_k:
            heapq.heappush(ranked_heap, entry)
        elif entry > ranked_heap[0]:
            heapq.heapreplace(ranked_heap, entry)

    ranked_ids = [
        pmc_id for _, _, pmc_id in sorted(ranked_heap, reverse=True)
    ]
    ranked_cases = [disease_rows[pmc_id] for pmc_id in ranked_ids]

    local_metadata = _fetch_local_metadata(ranked_ids, disease_rows)
    pmids_to_backfill = [
        local_metadata.get(pmc_id, {}).get("pmid")
        for pmc_id in ranked_ids
        if local_metadata.get(pmc_id, {}).get("pmid")
    ]
    pubmed_backfill = _fetch_pubmed_backfill(pmids_to_backfill, warnings)

    enriched_cases = []
    for case in ranked_cases:
        pmc_id = case["pmc_id"]
        metadata = local_metadata.get(pmc_id, {})
        combined = {
            "pmc_id": pmc_id,
            "pmcid": metadata.get("pmcid") or f"PMC{pmc_id}",
            "year": metadata.get("year") or case.get("year"),
            "diseases": case.get("diseases", []),
            "score": case.get("score"),
            "match_details": case.get("match_details", {}),
            "pmid": metadata.get("pmid"),
            "journal": metadata.get("journal"),
            "pub_date": metadata.get("pub_date"),
            "volume": metadata.get("volume"),
            "pages": metadata.get("pages"),
            "license": metadata.get("license"),
            "title": metadata.get("title"),
            "abstract": metadata.get("abstract"),
            "category": metadata.get("category"),
            "rarity_level": metadata.get("rarity_level"),
            "local_rarity_reasoning": metadata.get("reasoning"),
        }

        pubmed_row = pubmed_backfill.get(str(combined.get("pmid"))) if combined.get("pmid") else None
        if pubmed_row:
            combined["doi"] = pubmed_row.get("doi")
            combined["pmcid"] = combined.get("pmcid") or pubmed_row.get("pmcid")
            combined["title"] = combined.get("title") or pubmed_row.get("title")
            combined["abstract"] = combined.get("abstract") or pubmed_row.get("abstract")
            combined["journal"] = combined.get("journal") or pubmed_row.get("journal")
            combined["year"] = combined.get("year") or pubmed_row.get("year")
        else:
            combined["doi"] = None

        enriched_cases.append(combined)

    if fetch_full_text:
        for record in enriched_cases[: max(1, min(2, len(enriched_cases)) )]:
            try:
                record["full_text_excerpt"] = _fetch_full_text_snippet(record.get("pmcid") or record.get("pmc_id"))
            except Exception as exc:
                warnings.append(f"Full-text fetch failed for {record.get('pmcid') or record.get('pmid')}: {exc}")

    retrieved_records = [_project_retrieved_record(record) for record in enriched_cases]
    llm_candidates = retrieved_records[: min(6, len(retrieved_records))]

    llm_assessment = None
    if run_llm_filter and llm_candidates:
        llm_assessment = _run_llm_evidence_filter(
            query_bundle=query_bundle,
            candidates=llm_candidates,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            api_key_env=api_key_env,
            warnings=warnings,
        )

    candidate_map = {record["paper_id"]: record for record in retrieved_records}
    selected_papers = []

    if isinstance(llm_assessment, dict):
        for item in llm_assessment.get("selected_papers", []):
            paper_id = str(item.get("paper_id", "")).strip()
            record = candidate_map.get(paper_id)
            if not record:
                continue
            selected = dict(record)
            selected["relevance"] = _sanitize_level(item.get("relevance"), default="medium")
            selected["supports_claims"] = _coerce_claim_list(item.get("supports"))
            selected["selection_reason"] = _truncate(item.get("reason"), 320)
            selected_papers.append(selected)
        selected_papers = _dedupe_preserve_order(selected_papers)

    if not selected_papers:
        for record in retrieved_records[: min(4, len(retrieved_records))]:
            fallback_record = dict(record)
            fallback_record["relevance"] = "medium"
            fallback_record["supports_claims"] = []
            fallback_record["selection_reason"] = "Selected by retrieval score because LLM filtering was unavailable or returned no usable paper IDs."
            selected_papers.append(fallback_record)

    if llm_assessment:
        assessment = {
            "summary": _truncate(llm_assessment.get("summary"), 700),
            "novelty_level": _sanitize_level(llm_assessment.get("novelty_level"), default="medium"),
            "importance_level": _sanitize_level(llm_assessment.get("importance_level"), default="medium"),
            "supported_claim_candidates": _coerce_claim_list(llm_assessment.get("supported_claim_candidates")),
            "caveats": _coerce_claim_list(llm_assessment.get("caveats")),
            "uncertainty": _truncate(llm_assessment.get("uncertainty"), 260),
        }
    else:
        assessment = _fallback_assessment(query_bundle, selected_papers)

    citation_ready_dois = _dedupe_preserve_order(
        [paper.get("doi") for paper in selected_papers if paper.get("doi")]
    )

    papers_without_doi = [
        {
            "paper_id": paper.get("paper_id"),
            "pmid": paper.get("pmid"),
            "pmcid": paper.get("pmcid"),
            "title": paper.get("title"),
        }
        for paper in selected_papers
        if not paper.get("doi")
    ]

    result = {
        "current_case": {
            "pmc_id": query_bundle.get("current_pmc_id"),
            "pmid": query_bundle.get("current_pmid"),
            "title": query_bundle.get("source_metadata", {}).get("title"),
            "query_diseases": query_bundle.get("query_diseases", []),
            "query_phrases_used_for_retrieval": query_bundle.get("query_phrases", []),
            "related_keywords": query_bundle.get("related_keywords", []),
            "case_context_excerpt": _truncate(query_bundle.get("case_context"), 900),
        },
        "retrieved_similar_cases": retrieved_records,
        "top_relevant_papers": selected_papers,
        "citation_ready_dois": citation_ready_dois,
        "supporting_papers_missing_doi": papers_without_doi,
        "assessment": {
            **assessment,
            "analysis_method": "local_rag_plus_llm" if llm_assessment else "local_rag_with_heuristic_fallback",
            "downstream_citation_hint": (
                "Pass `citation_ready_dois` into `fetch_ama_citations(dois)` to format references."
            ),
        },
        "warnings": warnings,
    }

    return result
