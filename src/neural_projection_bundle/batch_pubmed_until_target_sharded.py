
import os, time, csv, argparse, json, random
from typing import List, Dict, Any, Tuple, Set
import requests

from openai import OpenAI
from prompts_llm import SYSTEM_PROMPT, USER_TEMPLATE, JSON_SCHEMA
from method_lexicon import infer_method
from doi_utils import resolve_best_doi

APP_NAME = "np-loop-llm"
HEADERS_JSON = {"Accept":"application/json"}

# ---------- Robust HTTP helpers ----------
def backoff_sleep(base=0.6, factor=1.8, jitter=0.25, attempt=0):
    import random, time
    t = base * (factor ** attempt) + random.uniform(0, jitter)
    time.sleep(min(t, 8.0))

def eutils_get(url: str, params: dict, max_attempts: int = 6):
    for att in range(max_attempts):
        try:
            r = requests.get(url, params=params, headers=HEADERS_JSON, timeout=40)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                backoff_sleep(attempt=att); continue
            r.raise_for_status()
            return r
        except Exception:
            backoff_sleep(attempt=att)
    return None

def safe_json(url: str, params: dict, attempts: int = 5, sleep_base: float = 0.6):
    for att in range(attempts):
        r = eutils_get(url, params)
        if r is None:
            time.sleep(sleep_base * (att + 1)); 
            continue
        ct = (r.headers.get("content-type") or "").lower()
        try:
            return r.json()
        except Exception:
            snippet = (r.text or "")[:180].replace("\n", "\\n")
            print(f"[safe_json] Non-JSON response (ct={ct}). retry {att+1}/{attempts}. head={snippet}")
            time.sleep(sleep_base * (att + 1))
            continue
    return {}

# ---------- PubMed utilities ----------
def esearch_count(query: str, email: str, api_key: str, mindate=None, maxdate=None, datetype="pdat") -> int:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db":"pubmed","term":query,"retmode":"json","retmax":0,"email":email,"tool":APP_NAME}
    if api_key: params["api_key"] = api_key
    if mindate and maxdate:
        params.update({"mindate": mindate, "maxdate": maxdate, "datetype": datetype})
    j = safe_json(url, params)
    return int(j.get("esearchresult",{}).get("count","0"))

def esearch_ids(query: str, email: str, api_key: str, retstart: int, retmax: int,
                mindate=None, maxdate=None, datetype="pdat") -> List[str]:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db":"pubmed","term":query,"retmode":"json","retmax":retmax,"retstart":retstart,"email":email,"tool":APP_NAME}
    if api_key: params["api_key"] = api_key
    if mindate and maxdate:
        params.update({"mindate": mindate, "maxdate": maxdate, "datetype": datetype})
    data = safe_json(url, params)
    return data.get("esearchresult",{}).get("idlist",[])

def esummary_details(ids: List[str], email: str, api_key: str) -> List[Dict[str,str]]:
    if not ids: return []
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    out = []
    for i in range(0, len(ids), 200):
        chunk = ",".join(ids[i:i+200])
        params = {"db":"pubmed","id":chunk,"retmode":"json","email":email,"tool":APP_NAME}
        if api_key: params["api_key"] = api_key
        res = safe_json(url, params).get("result",{})
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

def europepmc_links(pmid: str) -> Dict[str, str]:
    base = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": f"EXT_ID:{pmid} AND SRC:MED", "format": "json"}
    try:
        r = requests.get(base, params=params, timeout=20)
        if r.status_code != 200: return {}
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
    except Exception:
        return {}

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

def get_pmc_pdf_text(pmcid: str) -> str:
    try:
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf"
        r = requests.get(url, timeout=25)
        if r.status_code == 200 and r.headers.get("content-type","").lower().startswith("application/pdf"):
            return pdf_bytes_to_text(r.content)
    except Exception:
        pass
    return ""

def fetch_url_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=25)
        ct = r.headers.get("content-type","").lower()
        if "pdf" in ct: return pdf_bytes_to_text(r.content)
        if "html" in ct or "xml" in ct or url.endswith(".html"): return html_to_text(r.text)
    except Exception:
        return ""
    return ""

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
    # fallback: abstract only
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db":"pubmed","id":pmid,"retmode":"text","rettype":"abstract","email":email,"tool":APP_NAME}
    if api_key: params["api_key"] = api_key
    r = eutils_get(base, params)
    info["text"] = r.text if (r and r.status_code == 200) else ""
    return info

# ---------- Shard planner ----------
def plan_shards(query: str, email: str, api_key: str, start_year: int, end_year: int,
                cap: int = 9000) -> List[Tuple[int,int]]:
    """Recursively split [start_year, end_year] into shards with <= cap hits each."""
    def count_range(y0, y1):
        return esearch_count(query, email, api_key, mindate=str(y0), maxdate=str(y1), datetype="pdat")
    def split(y0, y1):
        c = count_range(y0, y1)
        if c <= cap:
            return [(y0, y1, c)]
        if y0 == y1:
            # one year still too big; split by months
            months = []
            for m0,m1 in [(1,6),(7,12)]:
                mindate=f"{y0}/{m0:02d}/01"; maxdate=f"{y0}/{m1:02d}/31"
                cm = esearch_count(query, email, api_key, mindate=mindate, maxdate=maxdate, datetype="pdat")
                if cm <= cap:
                    months.append((mindate, maxdate, cm))
                else:
                    # final fallback: quarter split
                    for q in [(1,3),(4,6),(7,9),(10,12)]:
                        mind=f"{y0}/{q[0]:02d}/01"; maxd=f"{y0}/{q[1]:02d}/31"
                        cq = esearch_count(query, email, api_key, mindate=mind, maxdate=maxd, datetype="pdat")
                        months.append((mind, maxd, cq))
            return months
        mid = (y0 + y1)//2
        left = split(y0, mid)
        right = split(mid+1, y1)
        return left + right
    raw = split(start_year, end_year)
    # Normalize: convert (year,year,count) to (mindate,maxdate,count) strings
    shards = []
    for a,b,c in raw:
        if isinstance(a,int) and isinstance(b,int):
            shards.append((f"{a}", f"{b}", c))
        else:
            # already 'YYYY/MM/DD' strings
            shards.append((a,b,c))
    return shards

# ---------- LLM extraction ----------
def openai_extract(text: str, meta: Dict[str,str], region_hints: str, model: str, temperature: float=0.0) -> list:
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
    kwargs = dict(
        model=model,
        response_format={"type":"json_schema","json_schema":JSON_SCHEMA},
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user}],
    )
    if temperature is not None and temperature != 1:
        try: kwargs["temperature"] = float(temperature)
        except Exception: pass
    resp = client.chat.completions.create(**kwargs)
    data = json.loads(resp.choices[0].message.content)
    return data.get("items", [])

# ---------- CSV helpers ----------
def ensure_header(path: str, header: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()

def count_rows(path: str) -> int:
    if not os.path.exists(path): return 0
    with open(path, "r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Sharded PubMed loop runner (bypass 10k cap by date shards).")
    ap.add_argument("--email", default=os.environ.get("NCBI_EMAIL",""), help="NCBI-required email")
    ap.add_argument("--api_key", default=os.environ.get("NCBI_API_KEY",""))
    ap.add_argument("--query", required=True)
    ap.add_argument("--out_csv", default="/content/out_pubmed_loop.csv")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--region_hints", default="M1,V1,CPu,MD,CA1,Thalamus,Cerebellum,STN,SNc,GPe,GPi,PPN,SMA,pre-SMA,PFC,Hippocampus,Putamen,Caudate")
    ap.add_argument("--target_rows", type=int, default=60100)
    ap.add_argument("--chunk_size", type=int, default=100)
    ap.add_argument("--state_json", default="/content/loop_state.json")
    ap.add_argument("--sleep_base", type=float, default=0.45)
    ap.add_argument("--year_start", type=int, default=1950)
    ap.add_argument("--year_end", type=int, default=2025)
    args = ap.parse_args()

    assert args.email, "NCBI email is required"

    header = ["sender","receiver","reference","journal","DOI","Taxon","Method","Pointer","Figure"]
    ensure_header(args.out_csv, header)

    # State
    state = {"current_shard": 0, "retstart": 0, "shards": [], "processed_pmids": []}
    if os.path.exists(args.state_json):
        try:
            state.update(json.load(open(args.state_json,"r",encoding="utf-8")))
        except Exception:
            pass

    if not state.get("shards"):
        shards = plan_shards(args.query, args.email, args.api_key, args.year_start, args.year_end, cap=9000)
        # keep only shards with >0 count
        state["shards"] = [s for s in shards if s[2] > 0]
        state["current_shard"] = 0
        state["retstart"] = 0
        with open(args.state_json,"w",encoding="utf-8") as sf:
            json.dump(state, sf, ensure_ascii=False, indent=2)
        print(f"[Plan] {len(state['shards'])} shards planned.")

    current_rows = count_rows(args.out_csv)
    print(f"[Init] CSV rows: {current_rows} / target {args.target_rows}")

    seen_keys: Set[Tuple[str,int,int,str]] = set()

    while current_rows < args.target_rows and state["current_shard"] < len(state["shards"]):
        mindate, maxdate, shard_count = state["shards"][state["current_shard"]]
        print(f"[Shard {state['current_shard']+1}/{len(state['shards'])}] range={mindate}..{maxdate} (count≈{shard_count}), retstart={state['retstart']}")

        # fetch PMID ids within shard (retstart guaranteed <10k due to cap)
        ids = esearch_ids(args.query, args.email, args.api_key, state["retstart"], args.chunk_size,
                          mindate=mindate, maxdate=maxdate, datetype="pdat")
        if not ids:
            # move to next shard
            state["current_shard"] += 1
            state["retstart"] = 0
            with open(args.state_json,"w",encoding="utf-8") as sf:
                json.dump(state, sf, ensure_ascii=False, indent=2)
            continue

        summaries = esummary_details(ids, args.email, args.api_key)
        with open(args.out_csv,"a",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            for meta in summaries:
                pmid = meta.get("pmid","")
                if pmid in state.get("processed_pmids", []):
                    continue

                info = resolve_text_for_pmid(pmid, args.email, args.api_key)
                text = info.get("text","")
                if not text:
                    state["processed_pmids"].append(pmid)
                    time.sleep(args.sleep_base); 
                    continue

                try:
                    meta["doi"] = resolve_best_doi(
                        pmid=pmid, email=args.email, api_key=args.api_key,
                        title=meta.get("title",""), journal=meta.get("journal",""),
                        year=meta.get("year",""), prefer=info.get("doi","") or meta.get("doi","")
                    )
                except Exception as e:
                    print(f"[PMID {pmid}] DOI resolve failed: {e}.")
                    meta["doi"] = info.get("doi","") or meta.get("doi","") or ""
                meta["pmcid"] = info.get("pmcid","")

                tl = text.lower()
                taxon = "Unspecified"
                if any(k in tl for k in ["rat","rats","rattus"]): taxon = "Rat"
                elif any(k in tl for k in ["mouse","mice","murine","mus musculus"]): taxon = "Mouse"
                elif any(k in tl for k in ["monkey","macaque","marmoset","nonhuman primate"]): taxon = "Non-human primate"
                elif any(k in tl for k in ["human","patients"]): taxon = "Human"

                try:
                    items = openai_extract(text, meta, args.region_hints, args.model, temperature=args.temperature)
                except Exception as e:
                    print(f"[PMID {pmid}] OpenAI failed: {e}")
                    state["processed_pmids"].append(pmid)
                    time.sleep(args.sleep_base)
                    continue

                ref = meta.get("title","").strip()
                for it in items:
                    st = it.get("offsets",{}).get("start",None)
                    en = it.get("offsets",{}).get("end",None)
                    ptr = it.get("pointer","") or ""
                    key = (pmid, int(st or -1), int(en or -1), ptr[:80])
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

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
                    current_rows += 1
                    if current_rows >= args.target_rows:
                        print(f"[Reached] {args.target_rows}.")
                        with open(args.state_json,"w",encoding="utf-8") as sf:
                            json.dump(state, sf, ensure_ascii=False, indent=2)
                        return

                state["processed_pmids"].append(pmid)
                time.sleep(args.sleep_base)

        # advance within shard
        state["retstart"] += args.chunk_size
        with open(args.state_json,"w",encoding="utf-8") as sf:
            json.dump(state, sf, ensure_ascii=False, indent=2)

    print(f"Finished. CSV at {args.out_csv}, state at {args.state_json}")

if __name__ == "__main__":
    main()
