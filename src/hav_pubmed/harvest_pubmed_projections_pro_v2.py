#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
harvest_pubmed_projections_pro_v2.py
- v2: Robust fallback when PubMed History (WebEnv/QueryKey) fails at ESummary
      ("Unable to obtain query #1" etc.). If history-based ESummary fails,
      we page IDs via ESearch (no history) for that window and call ESummary by IDs.
- Also explicitly sets retmode="xml" for ESummary calls.
- Everything else is identical to the previous "pro" script.

(See original pro script header for full feature list.)
"""

import csv
import os
import re
import sys
import time
import argparse
from typing import Dict, List, Tuple, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from xml.etree import ElementTree as ET
from Bio import Entrez

# ---------------------------
# Base query building
# ---------------------------

TRACER_TERMS_BASE = [
    'PHA-L', 'Phaseolus vulgaris leucoagglutinin', 'biocytin',
    'BDA', 'biotinylated dextran amine',
    'Fluoro-Gold', 'FG', 'Fast Blue',
    'DiI', 'DiO', 'DiA', 'carbocyanine dye',
    'CTB', 'cholera toxin subunit B', 'choleragenoid',
    'WGA-HRP', 'wheat germ agglutinin', 'horseradish peroxidase', 'HRP',
    'True Blue', 'Diamidino Yellow', 'Nuclear Yellow',
    'rabies virus', 'G-deleted rabies', 'PRV', 'pseudorabies', 'Bartha',
    'HSV-1', 'HSV1', 'H129', 'herpes simplex virus',
    'VSV', 'vesicular stomatitis virus', 'Sindbis',
    'AAV', 'adeno-associated virus', 'CAV-2', 'canine adenovirus 2',
    'lentivirus', 'retrograde AAV', 'anterograde AAV',
    'TRIO tracing', 'monosynaptic tracing', 'transsynaptic tracer',
    'CLARITY', 'iDISCO', 'uDISCO', 'SHIELD', 'MAP', 'expansion microscopy',
    'optogenetic', 'optogenetics', 'channelrhodopsin', 'ChR2', 'halorhodopsin', 'ArchT',
    'DTI', 'diffusion tensor imaging', 'tractography',
    'fMRI', 'functional MRI', 'resting-state fMRI', 'rs-fMRI', 'effective connectivity',
]

GENERIC_TERMS = r"""
(
  "Neural Pathways"[MeSH Terms] OR connectome[tiab] OR "functional connectivity"[tiab] OR "structural connectivity"[tiab]
  OR connectivity[tiab] OR "effective connectivity"[tiab]
  OR "diffusion tensor imaging"[tiab] OR DTI[tiab] OR tractography[tiab]
  OR "tract-tracing"[tiab] OR "tract tracing"[tiab] OR "anterograde tracer"[tiab] OR "retrograde tracer"[tiab]
  OR "axonal tracing"[tiab] OR "axon tracing"[tiab]
  OR projection[tiab] OR projections[tiab] OR "neural projection"[tiab] OR "neural projections"[tiab]
  OR pathway[tiab] OR pathways[tiab]
  OR optogenetic*[tiab] OR channelrhodopsin[tiab] OR optogenetics[tiab]
  OR fMRI[tiab] OR "functional MRI"[tiab] OR "resting-state fMRI"[tiab] OR rs-fMRI[tiab]
)
AND
(
  brain[MeSH Terms] OR brain[tiab] OR cortex[tiab] OR thalamus[tiab] OR cerebellum[tiab] OR hippocampus[tiab]
  OR basal ganglia[tiab] OR striatum[tiab] OR amygdala[tiab] OR hypothalamus[tiab] OR pallidum[tiab]
)
"""

def build_tracer_or_block(terms: List[str]) -> str:
    terms = [t.strip() for t in terms if t and str(t).strip()]
    joined = " OR ".join([f'{t}[tiab]' for t in terms])
    return f"({joined})" if joined else ""

def load_tracers_csv(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    terms = []
    with open(path, encoding="utf-8") as f:
        import csv as _csv
        reader = _csv.reader(f)
        for row in reader:
            row = [c.strip() for c in row if c and c.strip()]
            if not row:
                continue
            terms.append(row[0])
    return terms

DEFAULT_QUERY = f"({GENERIC_TERMS}) OR {build_tracer_or_block(TRACER_TERMS_BASE)}"

# ---------------------------
# HTTP helpers
# ---------------------------

def _json_or_none(resp: requests.Response) -> Optional[dict]:
    try:
        return resp.json()
    except Exception:
        return None

# ---------------------------
# Entrez helpers
# ---------------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def esearch_history(query: str):
    h = Entrez.esearch(db="pubmed", term=query, retmax=0, usehistory="y", retmode="xml")
    res = Entrez.read(h); h.close()
    return int(res["Count"]), res.get("WebEnv"), res.get("QueryKey")

@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def esummary_by_history(webenv: str, qk: str, retstart: int, retmax: int):
    h = Entrez.esummary(db="pubmed", webenv=webenv, query_key=qk, retstart=retstart, retmax=retmax, retmode="xml")
    records = Entrez.read(h); h.close()
    return records

@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def esearch_ids_window(query: str, retstart: int, retmax: int) -> List[str]:
    h = Entrez.esearch(db="pubmed", term=query, retstart=retstart, retmax=retmax, usehistory="n", retmode="xml")
    res = Entrez.read(h); h.close()
    return res.get("IdList", [])

@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def esummary_by_ids(id_list: List[str]):
    if not id_list:
        return []
    h = Entrez.esummary(db="pubmed", id=",".join(id_list), retmode="xml")
    records = Entrez.read(h); h.close()
    return records

@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def efetch_pubmed_xml(pmids: List[str]) -> str:
    h = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml")
    data = h.read(); h.close()
    return data

@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def efetch_pmc_xml(pmcid: str) -> str:
    h = Entrez.efetch(db="pmc", id=pmcid, rettype="xml", retmode="xml")
    data = h.read(); h.close()
    return data

# MeSH lookup
@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def mesh_lookup_descriptor(term: str) -> Tuple[str, List[str]]:
    h = Entrez.esearch(db="mesh", term=term, retmax=1, retmode="xml")
    res = Entrez.read(h); h.close()
    if not res.get("IdList"): return "", []
    mesh_id = res["IdList"][0]
    h2 = Entrez.efetch(db="mesh", id=mesh_id, rettype="xml", retmode="xml")
    xml = h2.read(); h2.close()
    try:
        root = ET.fromstring(xml)
        ui = ""
        tree_numbers = []
        for d in root.findall(".//DescriptorRecord"):
            ui_el = d.find("./DescriptorUI")
            if ui_el is not None and ui_el.text:
                ui = ui_el.text.strip()
            for tn in d.findall(".//TreeNumber"):
                if tn.text:
                    tree_numbers.append(tn.text.strip())
        return ui, sorted(set(tree_numbers))
    except ET.ParseError:
        return "", []

# Taxonomy lookup
@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def taxonomy_esearch_id(name: str) -> Optional[str]:
    h = Entrez.esearch(db="taxonomy", term=name, retmax=1, retmode="xml")
    res = Entrez.read(h); h.close()
    if res.get("IdList"):
        return res["IdList"][0]
    return None

# Europe PMC / Crossref
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CROSSREF_WORKS = "https://api.crossref.org/works/"

@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def europe_pmc_lookup_by_pmcid(pmcid: str) -> Tuple[Optional[str], Optional[str]]:
    params = {"query": f"PMCID:{pmcid}", "format": "json"}
    r = requests.get(EPMC_SEARCH, params=params, timeout=30)
    r.raise_for_status()
    j = _json_or_none(r)
    if not j or "resultList" not in j or "result" not in j["resultList"]:
        return None, None
    results = j["resultList"]["result"]
    if not results: return None, None
    rec = results[0]
    oa_url = None; lic = None
    if rec.get("fullTextUrlList") and "fullTextUrl" in rec["fullTextUrlList"]:
        for item in rec["fullTextUrlList"]["fullTextUrl"]:
            if item.get("availability") in ("Open access", "Open Access", "OA") or rec.get("isOpenAccess") == "Y":
                if item.get("url"):
                    oa_url = item["url"]; break
    if rec.get("license"): lic = rec["license"]
    return oa_url, lic

@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def crossref_lookup_by_doi(doi: str) -> Tuple[Optional[str], Optional[str]]:
    if not doi: return None, None
    url = CROSSREF_WORKS + requests.utils.quote(doi)
    r = requests.get(url, timeout=30, headers={"User-Agent": "WholeBIF-Harvester/1.0 (mailto:you@example.org)"})
    r.raise_for_status()
    j = _json_or_none(r)
    if not j or "message" not in j: return None, None
    msg = j["message"]
    pub_url = msg.get("URL")
    lic = None
    if msg.get("license"):
        lic = msg["license"][0].get("URL") or msg["license"][0].get("content-version")
    return pub_url, lic

# -------------- parsing / taxon --------------

SPECIES_HINTS = [
    "Homo sapiens", "human", "humans",
    "Mus musculus", "mouse", "mice",
    "Rattus", "rat", "rats",
    "Macaca", "macaque", "macaques",
    "Callithrix", "marmoset",
    "Cavia", "guinea pig",
    "Felis catus", "cat", "cats",
    "Canis familiaris", "dog", "dogs",
    "Mustela putorius furo", "ferret",
    "Sus scrofa", "pig", "pigs", "swine",
    "Ovis aries", "sheep",
    "Danio rerio", "zebrafish",
    "Drosophila melanogaster", "drosophila",
    "Caenorhabditis elegans", "c. elegans",
    "Gallus gallus", "chicken",
    "zebra finch", "Taeniopygia guttata",
    "nonhuman primate", "non-human primate",
]

SPECIES_REGEX = re.compile(r"\b(" + "|".join(sorted(map(re.escape, SPECIES_HINTS), key=len, reverse=True)) + r")\b", re.IGNORECASE)

def norm_space(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def dedup_stable(seq: List[str]) -> List[str]:
    out, seen = [], set()
    for x in seq:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def parse_pubmed_xml(xml_text: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    root = ET.fromstring(xml_text)
    for art in root.findall("./PubmedArticle"):
        pmid_el = art.find("./MedlineCitation/PMID")
        if pmid_el is None or not pmid_el.text: continue
        pmid = pmid_el.text.strip()
        abs_parts = []
        for abst in art.findall("./MedlineCitation/Article/Abstract/AbstractText"):
            txt = "".join(abst.itertext()).strip()
            if txt:
                label = abst.get("Label")
                abs_parts.append(f"{label}: {txt}" if label else txt)
        abstract = norm_space(" ".join(abs_parts)) if abs_parts else ""
        mesh_terms = []; mesh_desc_only = []
        for mh in art.findall("./MedlineCitation/MeshHeadingList/MeshHeading"):
            d = mh.find("./DescriptorName")
            if d is not None and d.text:
                dname = d.text.strip()
                mesh_desc_only.append(dname); mesh_terms.append(dname)
            for q in mh.findall("./QualifierName"):
                if q is not None and q.text:
                    mesh_terms.append(q.text.strip())
        keywords = []
        for kwl in art.findall("./MedlineCitation/KeywordList"):
            for kw in kwl.findall("./Keyword"):
                if kw.text: keywords.append(kw.text.strip())
        pmcid = ""
        for aid in art.findall("./PubmedData/ArticleIdList/ArticleId"):
            idtype = (aid.get("IdType") or "").lower()
            if idtype == "pmc" and aid.text:
                val = aid.text.strip()
                pmcid = val if val.upper().startswith("PMC") else f"PMC{val}"
                break
        out[pmid] = {
            "abstract": abstract,
            "mesh_terms": mesh_terms,
            "mesh_desc_only": mesh_desc_only,
            "keywords": keywords,
            "pmcid": pmcid,
        }
    return out

def detect_taxa_and_ids(mesh_desc_only: List[str], keywords: List[str], abstract: str) -> Tuple[List[str], List[str], str]:
    def try_extract(texts: List[str]) -> List[str]:
        hits = []
        for t in texts:
            for m in SPECIES_REGEX.finditer(t or ""):
                hits.append(m.group(1))
        return dedup_stable(hits)
    ms = try_extract(mesh_desc_only); source = ""
    if ms:
        source = "MeSH"; labels = ms
    else:
        ks = try_extract(keywords)
        if ks: source = "Keyword"; labels = ks
        else:
            as_ = try_extract([abstract])
            if as_: source = "Abstract"; labels = as_
            else: return [], [], ""
    tax_ids = []
    for lab in labels:
        tid = taxonomy_esearch_id(lab)
        if tid: tax_ids.append(tid)
    return labels, tax_ids, source

# -------------- CSV helpers --------------

def get_existing_pmids(out_csv: str) -> set:
    if not os.path.exists(out_csv): return set()
    pmids = set()
    with open(out_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = (row.get("pmid") or "").strip()
            if pmid: pmids.add(pmid)
    return pmids

def ensure_headers(out_csv: str, fulltext_cols: bool = True):
    need_header = not os.path.exists(out_csv)
    f = open(out_csv, "a", newline="", encoding="utf-8")
    fields = [
        "title","doi","pmid","journal","abstract",
        "taxon_labels","taxon_ncbi_ids","taxon_source",
        "pmcid","oa_url","oa_license","publisher_url","publisher_license",
        "mesh_ui","mesh_tree_numbers"
    ]
    if fulltext_cols:
        fields += ["fulltext_path","fulltext_chars"]
    writer = csv.DictWriter(f, fieldnames=fields)
    if need_header: writer.writeheader()
    return f, writer

# -------------- MeSH UI/Tree enrich --------------

def enrich_mesh_fields(mesh_desc_only: List[str], sleep_sec: float, max_terms: int = 10) -> Tuple[str, str]:
    ui_map: Dict[str, str] = {}
    tree_map: Dict[str, List[str]] = {}
    for dname in mesh_desc_only[:max_terms]:
        try:
            ui, trees = mesh_lookup_descriptor(dname)
            if ui: ui_map[dname] = ui
            if trees: tree_map[dname] = trees
            time.sleep(sleep_sec)
        except Exception:
            pass
    mesh_ui_field = "; ".join(f"{k}:{v}" for k,v in ui_map.items()) if ui_map else ""
    mesh_trees_field = "; ".join(f"{k}:[{','.join(v)}]" for k,v in tree_map.items()) if tree_map else ""
    return mesh_ui_field, mesh_trees_field

# -------------- Main --------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--query", default=None)
    ap.add_argument("--tracers_csv", default=None)
    ap.add_argument("--chunk", type=int, default=200)   # v2: safer default
    ap.add_argument("--sleep", type=float, default=0.4) # v2: gentler default
    ap.add_argument("--max_count", type=int, default=None)
    ap.add_argument("--fetch_fulltext", action="store_true")
    ap.add_argument("--fulltext_dir", default=None)
    ap.add_argument("--fulltext_max_chars", type=int, default=3000)
    args = ap.parse_args()

    Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key

    # Build query
    if args.query:
        query = args.query
    else:
        tracer_terms = list(TRACER_TERMS_BASE)
        extra = load_tracers_csv(args.tracers_csv) if args.tracers_csv else []
        tracer_terms.extend(extra)
        # de-dup
        seen = set(); tracer_terms_dedup = []
        for t in tracer_terms:
            if t not in seen:
                seen.add(t); tracer_terms_dedup.append(t)
        query = f"({GENERIC_TERMS})"
        tracer_block = build_tracer_or_block(tracer_terms_dedup)
        if tracer_block:
            query = f"({query}) OR {tracer_block}"

    if args.fetch_fulltext and args.fulltext_dir:
        os.makedirs(args.fulltext_dir, exist_ok=True)

    existing_pmids = get_existing_pmids(args.out_csv)
    f, writer = ensure_headers(args.out_csv, fulltext_cols=True)

    try:
        print("Running ESearch (history)…", file=sys.stderr)
        total, webenv, qk = esearch_history(query)
        print(f"Total hits: {total}", file=sys.stderr)
        if total == 0:
            print("No results. Check query.", file=sys.stderr); return

        processed = 0
        retstart = 0
        chunk = args.chunk
        pbar_total = args.max_count if args.max_count is not None else total

        with tqdm(total=pbar_total, desc="Harvesting", unit="rec") as pbar:
            while retstart < total:
                if args.max_count is not None and processed >= args.max_count:
                    break

                # --- Robust summary fetch ---
                try:
                    summaries = esummary_by_history(webenv, qk, retstart, chunk)
                except Exception as e:
                    # Fallback: page IDs then summary-by-IDs
                    id_list = esearch_ids_window(query, retstart, chunk)
                    if not id_list:
                        retstart += chunk
                        continue
                    summaries = esummary_by_ids(id_list)

                time.sleep(args.sleep)

                page_pmids: List[str] = []
                esum_by_pmid: Dict[str, dict] = {}
                for rec in summaries:
                    pmid = rec.get("Id")
                    if not pmid: continue
                    if pmid in existing_pmids: continue
                    esum_by_pmid[pmid] = rec
                    page_pmids.append(pmid)

                # PubMed XML for details
                xml_info: Dict[str, dict] = {}
                if page_pmids:
                    for i in range(0, len(page_pmids), chunk):
                        sub = page_pmids[i:i+chunk]
                        xml = efetch_pubmed_xml(sub)
                        time.sleep(args.sleep)
                        xml_info.update(parse_pubmed_xml(xml))

                for pmid in page_pmids:
                    rec = esum_by_pmid.get(pmid, {})
                    if not rec: continue
                    title = rec.get("Title") or ""
                    if isinstance(title, list): title = title[0].get("content") if title else ""
                    title = norm_space(str(title))
                    journal = norm_space(rec.get("FullJournalName") or rec.get("Source") or "")
                    # DOI detection
                    doi = ""
                    if "DOI" in rec:
                        doi = (rec["DOI"] or "").strip()
                    else:
                        for item in rec.get("ArticleIds", []):
                            if isinstance(item, dict) and item.get("Name", "").lower() == "doi":
                                doi = (item.get("Value") or "").strip(); 
                                if doi: break
                        if not doi and "ELocationID" in rec:
                            val = rec["ELocationID"]
                            if isinstance(val, list):
                                for v in val:
                                    if isinstance(v, dict) and (v.get("EIdType","").lower()=="doi"):
                                        cand = (v.get("content") or "").strip()
                                        if cand: doi = cand; break
                    doi_link = f"https://doi.org/{doi}" if doi else ""

                    x = xml_info.get(pmid, {})
                    abstract = x.get("abstract","")
                    mesh_desc_only = x.get("mesh_desc_only", [])
                    keywords = x.get("keywords", [])
                    pmcid = x.get("pmcid","")

                    # Taxonomy
                    tax_labels, tax_ids, tax_source = detect_taxa_and_ids(mesh_desc_only, keywords, abstract)
                    taxon_labels = "; ".join(tax_labels) if tax_labels else ""
                    taxon_ncbi_ids = "; ".join(tax_ids) if tax_ids else ""

                    # MeSH enrich
                    mesh_ui_field, mesh_trees_field = enrich_mesh_fields(mesh_desc_only, args.sleep, max_terms=10)

                    # OA/publisher URLs
                    oa_url = ""; oa_license = ""
                    if pmcid:
                        try:
                            _oa, _lic = europe_pmc_lookup_by_pmcid(pmcid)
                            if _oa: oa_url = _oa
                            if _lic: oa_license = _lic
                            time.sleep(args.sleep)
                        except Exception:
                            pass

                    publisher_url = ""; publisher_license = ""
                    if doi:
                        try:
                            _purl, _plic = crossref_lookup_by_doi(doi)
                            if _purl: publisher_url = _purl
                            if _plic: publisher_license = _plic
                            time.sleep(args.sleep)
                        except Exception:
                            pass

                    # Fulltext (optional)
                    fulltext_path = ""; fulltext_chars = ""
                    if args.fetch_fulltext and pmcid:
                        try:
                            nxml = efetch_pmc_xml(pmcid)
                            time.sleep(args.sleep)
                            # parse
                            pieces = []
                            try:
                                root = ET.fromstring(nxml)
                                for abst in root.findall(".//abstract"):
                                    for p in abst.findall(".//p"):
                                        t = norm_space("".join(p.itertext()))
                                        if t: pieces.append(t)
                                for body in root.findall(".//body"):
                                    for p in body.findall(".//p"):
                                        t = norm_space("".join(p.itertext()))
                                        if t: pieces.append(t)
                                ft = "\n\n".join(pieces).strip()
                            except ET.ParseError:
                                ft = ""
                            if ft:
                                fulltext_chars = ft[:args.fulltext_max_chars]
                                if args.fulltext_dir:
                                    outp = os.path.join(args.fulltext_dir, f"{pmcid}.txt")
                                    with open(outp, "w", encoding="utf-8") as g:
                                        g.write(ft)
                                    fulltext_path = outp
                        except Exception:
                            pass

                    writer.writerow({
                        "title": title,
                        "doi": doi_link,
                        "pmid": pmid,
                        "journal": journal,
                        "abstract": abstract,
                        "taxon_labels": taxon_labels,
                        "taxon_ncbi_ids": taxon_ncbi_ids,
                        "taxon_source": tax_source,
                        "pmcid": pmcid,
                        "oa_url": oa_url,
                        "oa_license": oa_license,
                        "publisher_url": publisher_url,
                        "publisher_license": publisher_license,
                        "mesh_ui": mesh_ui_field,
                        "mesh_tree_numbers": mesh_trees_field,
                        "fulltext_path": fulltext_path,
                        "fulltext_chars": fulltext_chars
                    })
                    f.flush()
                    existing_pmids.add(pmid)
                    processed += 1
                    pbar.update(1)
                    if args.max_count is not None and processed >= args.max_count:
                        break

                retstart += chunk

    finally:
        f.close()

if __name__ == "__main__":
    main()
