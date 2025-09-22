
import os, time, csv, argparse, json, random
from typing import List, Dict, Any
import requests
from openai import OpenAI
from prompts_llm import SYSTEM_PROMPT, USER_TEMPLATE, JSON_SCHEMA
from method_lexicon import infer_method
from doi_utils import resolve_best_doi

def openai_extract(text: str, meta: Dict[str, str], region_hints: str, model: str) -> list:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)
    user = USER_TEMPLATE.format(
        title=meta.get("title",""), journal=meta.get("journal",""),
        year=meta.get("year",""), doi=meta.get("doi",""),
        pmid=meta.get("pmid",""), pmcid=meta.get("pmcid",""),
        region_hints=region_hints, text=text[:18000]
    )
    resp = client.chat.completions.create(
        model=model, temperature=0.0,
        response_format={"type":"json_schema","json_schema":JSON_SCHEMA},
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user}],
    )
    data = json.loads(resp.choices[0].message.content)
    return data.get("items", [])

def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    try:
        from io import BytesIO
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams
        laparams = LAParams(line_margin=0.2, word_margin=0.1, char_margin=2.0)
        return extract_text(BytesIO(pdf_bytes), laparams=laparams) or ""
    except Exception:
        return ""

def html_to_text(html: str) -> str:
    import re
    from html import unescape
    html = re.sub(r'<(script|style)[\s\S]*?</\1>', ' ', html, flags=re.I)
    html = re.sub(r'</?(br|p|div|li|tr|h\d)>', '\n', html, flags=re.I)
    text = re.sub(r'<[^>]+>', ' ', html)
    text = unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def europepmc_links(pmid: str) -> Dict[str, str]:
    base = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": f"EXT_ID:{pmid} AND SRC:MED", "format": "json"}
    r = requests.get(base, params=params, timeout=20)
    if r.status_code != 200:
        return {}
    links = {}
    for hit in r.json().get("resultList", {}).get("result", []):
        if "pmcid" in hit: links["pmcid"] = hit["pmcid"]
        if "doi" in hit: links["doi"] = hit["doi"]
        if "fullTextUrlList" in hit:
            for u in hit["fullTextUrlList"]["fullTextUrl"]:
                url = u.get("url",""); style = u.get("documentStyle",""); avail = u.get("availability","")
                if url.endswith(".pdf") or "pdf" in (style+avail).lower():
                    links["pdf"] = url
                elif url:
                    links["html"] = url
    return links

def get_pmc_pdf_text(pmcid: str) -> str:
    try:
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf"
        r = requests.get(url, timeout=20)
        if r.status_code == 200 and r.headers.get("content-type","").lower().startswith("application/pdf"):
            return pdf_bytes_to_text(r.content)
    except Exception:
        pass
    return ""

def fetch_url_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=20)
        ct = r.headers.get("content-type","").lower()
        if "pdf" in ct: return pdf_bytes_to_text(r.content)
        if "html" in ct or "xml" in ct or url.endswith(".html"): return html_to_text(r.text)
    except Exception:
        return ""
    return ""

def backoff_sleep(base=0.6, factor=1.8, jitter=0.2, attempt=0):
    t = base * (factor ** attempt) + random.uniform(0, jitter)
    time.sleep(t)

def eutils_get(url: str, params: dict, max_attempts: int = 5):
    for att in range(max_attempts):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200: return r
            if r.status_code in (429, 500, 502, 503, 504):
                backoff_sleep(attempt=att); continue
            r.raise_for_status(); return r
        except Exception:
            backoff_sleep(attempt=att)
    raise RuntimeError(f"E-utilities request failed after {max_attempts} attempts: {url}")

def pubmed_search_ids(query: str, email: str, api_key: str, retmax: int = 10) -> List[str]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db":"pubmed","term":query,"retmax":retmax,"retmode":"json","email":email,"tool":"np-batch-llm"}
    if api_key: params["api_key"] = api_key
    r = eutils_get(base, params)
    data = r.json()
    return data.get("esearchresult",{}).get("idlist",[])[:retmax]

def pubmed_summaries(ids: List[str], email: str, api_key: str) -> List[Dict[str, str]]:
    if not ids: return []
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    out = []
    for i in range(0, len(ids), 200):
        chunk = ",".join(ids[i:i+200])
        params = {"db":"pubmed","id":chunk,"retmode":"json","email":email,"tool":"np-batch-llm"}
        if api_key: params["api_key"] = api_key
        r = eutils_get(base, params)
        res = r.json().get("result",{})
        for k,v in res.items():
            if k == "uids": continue
            out.append({
                "pmid": k,
                "title": v.get("title",""),
                "journal": v.get("fulljournalname",""),
                "year": (v.get("pubdate","") or "").split(" ")[0],
                "doi": "",
            })
    return out

def efetch_abstract(pmid: str, email: str, api_key: str) -> str:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db":"pubmed","id":pmid,"retmode":"text","rettype":"abstract","email":email,"tool":"np-batch-llm"}
    if api_key: params["api_key"] = api_key
    r = eutils_get(base, params)
    return r.text if r.status_code == 200 else ""

def resolve_text_for_pmid(pmid: str, email: str, api_key: str) -> Dict[str, str]:
    info = {"text":"", "doi":"", "pmcid":""}
    links = europepmc_links(pmid)
    if "pmcid" in links:
        t = get_pmc_pdf_text(links["pmcid"])
        if t:
            info.update({"text":t,"pmcid":links["pmcid"],"doi":links.get("doi","")}); return info
    for k in ["pdf","html"]:
        if links.get(k):
            t = fetch_url_text(links[k])
            if t:
                info.update({"text":t,"pmcid":links.get("pmcid",""),"doi":links.get("doi","")}); return info
    info["text"] = efetch_abstract(pmid, email, api_key)
    return info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default=os.environ.get("NCBI_EMAIL",""), help="Your email (NCBI requirement)")
    ap.add_argument("--api_key", default=os.environ.get("NCBI_API_KEY",""), help="NCBI E-utilities API key")
    ap.add_argument("--query", default='(projection OR projects to OR innervat* OR afferent* OR efferent* OR input* OR output* OR connect*) AND (thalamus OR cortex OR striatum OR caudate OR putamen OR hippocampus OR "area V1" OR "area M1" OR cerebell*)')
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--out_csv", default="/content/out_pubmed10.csv")
    ap.add_argument("--model", default="gpt-5-nano")
    args = ap.parse_args()

    assert args.email, "NCBI email is required. Pass --email or set env NCBI_EMAIL"

    ids = pubmed_search_ids(args.query, args.email, args.api_key, retmax=args.count)
    summaries = pubmed_summaries(ids, args.email, args.api_key)

    header = ["sender","receiver","reference","journal","DOI","Taxon","Method","Pointer","Figure"]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader()

        region_hints = "M1,V1,CPu,MD,CA1,Thalamus,Cerebellum,STN,SNc,GPe,GPi,PPN,SMA,pre-SMA,PFC,Hippocampus,Putamen,Caudate"

        for meta in summaries:
            pmid = meta.get("pmid","")
            print(f"[PMID {pmid}] Resolving text...")
            info = resolve_text_for_pmid(pmid, args.email, args.api_key)
            text = info.get("text","")
            if not text:
                print(f"[PMID {pmid}] No text, skipping."); continue
            # Robust DOI resolution
            meta["doi"] = resolve_best_doi(
                pmid=pmid, email=args.email, api_key=args.api_key,
                title=meta.get("title",""), journal=meta.get("journal",""),
                year=meta.get("year",""), prefer=info.get("doi","") or meta.get("doi","")
            )
            meta["pmcid"] = info.get("pmcid","")

            # Coarse taxon
            tl = text.lower()
            taxon = "Unspecified"
            if any(k in tl for k in ["rat","rats","rattus"]): taxon = "Rat"
            elif any(k in tl for k in ["mouse","mice","murine","mus musculus"]): taxon = "Mouse"
            elif any(k in tl for k in ["monkey","macaque","marmoset","nonhuman primate"]): taxon = "Non-human primate"
            elif any(k in tl for k in ["human","patients"]): taxon = "Human"

            print(f"[PMID {pmid}] Calling OpenAI...")
            try:
                items = openai_extract(text, meta, region_hints, args.model)
            except Exception as e:
                print(f"[PMID {pmid}] OpenAI failed: {e}"); continue

            ref = f"{meta.get('title','')}".strip()
            wrote = 0
            for it in items:
                st = it.get("offsets",{}).get("start",None)
                en = it.get("offsets",{}).get("end",None)
                method_rule = infer_method(text, st, en)
                method_llm  = it.get("method_hint","") or "Unspecified"
                method = method_rule if method_rule != "Unspecified" else method_llm

                w.writerow({
                    "sender": it.get("sender",""),
                    "receiver": it.get("receiver",""),
                    "reference": ref,
                    "journal": meta.get("journal",""),
                    "DOI": meta.get("doi",""),
                    "Taxon": taxon,
                    "Method": method,
                    "Pointer": it.get("pointer",""),
                    "Figure": ", ".join(it.get("figure_ids",[]) or ([it.get("figure","")] if it.get("figure") else [])),
                })
                wrote += 1
            print(f"[PMID {pmid}] Wrote {wrote} rows.")
            time.sleep(0.15 if args.api_key else 0.45)

    print(f"Done. Wrote CSV to {args.out_csv}")

if __name__ == "__main__":
    main()
