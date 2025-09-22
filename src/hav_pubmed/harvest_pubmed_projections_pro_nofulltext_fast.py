#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
harvest_pubmed_projections_pro_nofulltext_fast.py

超高速化ポイント（--fast 時）:
- Europe PMC / Crossref 照会を完全スキップ
- MeSH UI / Tree のルックアップをスキップ
- Taxonomy ID 解決をスキップ（ラベル抽出のみ）
- PubMed History(WebEnv) を使わず、ESearch で ID をページング → ESummary & EFetch by IDs
- 既定: chunk=1000, sleep=0.1（API key がある前提。混雑時は上げてください）

通常モード（--fast なし）:
- 以前の nofulltext 版と同等（EuropePMC/Crossref/MeSH enrich/TaxID 有効）
- ただし、TaxID/MeSH の結果はメモリキャッシュして重複呼び出しを削減

出力列は nofulltext と同一:
title,doi,pmid,journal,abstract,
taxon_labels,taxon_ncbi_ids,taxon_source,
pmcid,oa_url,oa_license,publisher_url,publisher_license,
mesh_ui,mesh_tree_numbers
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
def esearch_ids_window(query: str, retstart: int, retmax: int) -> List[str]:
    h = Entrez.esearch(db="pubmed", term=query, retstart=retstart, retmax=retmax, usehistory="n", retmode="xml")
    res = Entrez.read(h); h.close()
    return res.get("IdList", [])

@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def esummary_by_history(webenv: str, qk: str, retstart: int, retmax: int):
    h = Entrez.esummary(db="pubmed", webenv=webenv, query_key=qk, retstart=retstart, retmax=retmax, retmode="xml")
    records = Entrez.read(h); h.close()
    return records

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

# MeSH lookup cache
_MESH_CACHE: Dict[str, Tuple[str, List[str]]] = {}
@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def mesh_lookup_descriptor(term: str) -> Tuple[str, List[str]]:
    if term in _MESH_CACHE:
        return _MESH_CACHE[term]
    h = Entrez.esearch(db="mesh", term=term, retmax=1, retmode="xml")
    res = Entrez.read(h); h.close()
    if not res.get("IdList"): 
        _MESH_CACHE[term] = ("", []); 
        return "", []
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
        _MESH_CACHE[term] = (ui, sorted(set(tree_numbers)))
        return _MESH_CACHE[term]
    except ET.ParseError:
        _MESH_CACHE[term] = ("", [])
        return "", []

# Taxonomy lookup cache
_TAX_CACHE: Dict[str, Optional[str]] = {}
@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def taxonomy_esearch_id(name: str) -> Optional[str]:
    if name in _TAX_CACHE:
        return _TAX_CACHE[name]
    h = Entrez.esearch(db="taxonomy", term=name, retmax=1, retmode="xml")
    res = Entrez.read(h); h.close()
    tid = res["IdList"][0] if res.get("IdList") else None
    _TAX_CACHE[name] = tid
    return tid

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

def detect_taxa_and_ids(mesh_desc_only: List[str], keywords: List[str], abstract: str, want_tax_ids: bool) -> Tuple[List[str], List[str], str]:
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
    if want_tax_ids:
        for lab in labels:
            tid = taxonomy_esearch_id(lab)
            if tid: tax_ids.append(tid)
    return labels, tax_ids, source

# -------------- CSV helpers --------------

DEFAULT_FIELDS_NOFT = [
    "title","doi","pmid","journal","abstract",
    "taxon_labels","taxon_ncbi_ids","taxon_source",
    "pmcid","oa_url","oa_license","publisher_url","publisher_license",
    "mesh_ui","mesh_tree_numbers"
]

def detect_existing_header(out_csv: str) -> Optional[List[str]]:
    if not os.path.exists(out_csv):
        return None
    try:
        with open(out_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return reader.fieldnames
    except Exception:
        return None

def open_writer_with_compat(out_csv: str):
    existing = detect_existing_header(out_csv)
    if existing and len(existing) > 0:
        fields = existing
        f = open(out_csv, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=fields)
        return f, writer, fields
    else:
        f = open(out_csv, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=list(DEFAULT_FIELDS_NOFT))
        writer.writeheader()
        return f, writer, list(DEFAULT_FIELDS_NOFT)

# -------------- MeSH UI/Tree enrich --------------

def enrich_mesh_fields(mesh_desc_only: List[str], sleep_sec: float, max_terms: int = 10, enable: bool = True) -> Tuple[str, str]:
    if not enable:
        return "", ""
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
    ap.add_argument("--chunk", type=int, default=None)  # will set per-mode
    ap.add_argument("--sleep", type=float, default=None)
    ap.add_argument("--max_count", type=int, default=None)
    ap.add_argument("--fast", action="store_true", help="極力速く（OA/Crossref/MeSH enrich/TaxIDをスキップし、IDページング方式に）")
    ap.add_argument("--no_history", action="store_true", help="履歴(WebEnv)を使わず ID ページングする")
    ap.add_argument("--no_epmc", action="store_true", help="EuropePMC 照会を行わない")
    ap.add_argument("--no_crossref", action="store_true", help="Crossref 照会を行わない")
    ap.add_argument("--no_mesh_enrich", action="store_true", help="MeSH UI/Tree 付与を行わない")
    ap.add_argument("--no_taxonomy_ids", action="store_true", help="Taxonomy ID 解決を行わない（ラベル抽出のみ）")
    args = ap.parse_args()

    # Mode defaults
    if args.fast:
        if args.chunk is None: args.chunk = 1000
        if args.sleep is None: args.sleep = 0.1
        args.no_history = True
        args.no_epmc = True
        args.no_crossref = True
        args.no_mesh_enrich = True
        args.no_taxonomy_ids = True
    else:
        if args.chunk is None: args.chunk = 300
        if args.sleep is None: args.sleep = 0.3

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
        seen = set(); tracer_terms_dedup = []
        for t in tracer_terms:
            if t not in seen:
                seen.add(t); tracer_terms_dedup.append(t)
        query = f"({GENERIC_TERMS})"
        tracer_block = build_tracer_or_block(tracer_terms_dedup)
        if tracer_block:
            query = f"({query}) OR {tracer_block}"

    # Resume-friendly writer
    existing_pmids = set()
    if os.path.exists(args.out_csv):
        with open(args.out_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pmid = (row.get("pmid") or "").strip()
                if pmid: existing_pmids.add(pmid)

    f, writer, fieldnames = open_writer_with_compat(args.out_csv)

    try:
        # Determine total via ESearch (history or not)
        print("Counting with ESearch…", file=sys.stderr)
        total, webenv, qk = 0, None, None
        if not args.no_history:
            total, webenv, qk = esearch_history(query)
        else:
            # Count via a small ESearch without history (retmax=0 still returns Count)
            h = Entrez.esearch(db="pubmed", term=query, retmax=0, usehistory="n", retmode="xml")
            res = Entrez.read(h); h.close()
            total = int(res["Count"])
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

                # Summaries obtain
                if not args.no_history and webenv and qk:
                    try:
                        summaries = esummary_by_history(webenv, qk, retstart, chunk)
                    except Exception:
                        # Fallback to ID window
                        id_list = esearch_ids_window(query, retstart, chunk)
                        summaries = esummary_by_ids(id_list)
                else:
                    id_list = esearch_ids_window(query, retstart, chunk)
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
                                doi = (item.get("Value") or "").strip()
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
                    tax_labels, tax_ids, tax_source = detect_taxa_and_ids(
                        mesh_desc_only, keywords, abstract, want_tax_ids=not args.no_taxonomy_ids
                    )
                    taxon_labels = "; ".join(tax_labels) if tax_labels else ""
                    taxon_ncbi_ids = "; ".join(tax_ids) if tax_ids else ""

                    # MeSH enrich (UI/Tree)
                    mesh_ui_field, mesh_trees_field = enrich_mesh_fields(
                        mesh_desc_only, args.sleep, max_terms=10, enable=not args.no_mesh_enrich
                    )

                    # OA/publisher URLs (skippable)
                    oa_url = ""; oa_license = ""
                    if pmcid and not args.no_epmc:
                        try:
                            _oa, _lic = europe_pmc_lookup_by_pmcid(pmcid)
                            if _oa: oa_url = _oa
                            if _lic: oa_license = _lic
                            time.sleep(args.sleep)
                        except Exception:
                            pass

                    publisher_url = ""; publisher_license = ""
                    if doi and not args.no_crossref:
                        try:
                            _purl, _plic = crossref_lookup_by_doi(doi)
                            if _purl: publisher_url = _purl
                            if _plic: publisher_license = _plic
                            time.sleep(args.sleep)
                        except Exception:
                            pass

                    row = {
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
                    }
                    aligned = {k: row.get(k, "") for k in fieldnames}

                    writer.writerow(aligned)
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
