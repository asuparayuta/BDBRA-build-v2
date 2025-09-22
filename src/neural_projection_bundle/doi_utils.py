
import re
import requests
from xml.etree import ElementTree as ET

DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.I)

def normalize_doi(doi: str) -> str:
    if not doi: return ""
    doi = doi.strip()
    doi = doi.replace("https://doi.org/","").replace("http://doi.org/","").replace("doi:","").strip()
    m = DOI_RE.search(doi)
    return m.group(0) if m else ""

def doi_from_text(text: str) -> str:
    if not text: return ""
    m = DOI_RE.search(text)
    return normalize_doi(m.group(0)) if m else ""

def pubmed_doi_via_esummary(pmid: str, email: str, api_key: str="") -> str:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {"db":"pubmed","id":pmid,"retmode":"json","email":email,"tool":"np-batch-llm"}
    if api_key: params["api_key"] = api_key
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        return ""
    res = r.json().get("result",{}).get(pmid,{})
    ids = res.get("articleids", [])
    for rec in ids:
        if rec.get("idtype","").lower()=="doi":
            return normalize_doi(rec.get("value",""))
    blob = " ".join(str(v) for v in res.values())
    return doi_from_text(blob)

def pubmed_doi_via_efetch_xml(pmid: str, email: str, api_key: str="") -> str:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db":"pubmed","id":pmid,"retmode":"xml","email":email,"tool":"np-batch-llm"}
    if api_key: params["api_key"] = api_key
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        return ""
    try:
        root = ET.fromstring(r.text)
        for art in root.iterfind(".//ArticleIdList/ArticleId"):
            if art.attrib.get("IdType","").lower() == "doi":
                return normalize_doi(art.text or "")
    except Exception:
        return ""
    return ""

def crossref_doi_by_biblio(title: str, journal: str, year: str) -> str:
    if not title:
        return ""
    try:
        url = "https://api.crossref.org/works"
        params = {"query.bibliographic": f"{title} {journal} {year}".strip(), "rows": 3}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200: return ""
        items = r.json().get("message",{}).get("items",[])
        for it in items:
            doi = normalize_doi(it.get("DOI",""))
            if doi: return doi
    except Exception:
        return ""
    return ""

def resolve_best_doi(pmid: str, email: str, api_key: str, title: str, journal: str, year: str, prefer: str="") -> str:
    d = normalize_doi(prefer)
    if d: return d
    d = pubmed_doi_via_esummary(pmid, email, api_key)
    if d: return d
    d = pubmed_doi_via_efetch_xml(pmid, email, api_key)
    if d: return d
    d = crossref_doi_by_biblio(title, journal, year)
    return d or ""
