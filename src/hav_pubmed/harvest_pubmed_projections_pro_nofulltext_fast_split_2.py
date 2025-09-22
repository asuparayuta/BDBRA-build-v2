#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harvest_pubmed_projections_pro_nofulltext_fast_split_v3.py

変更点（要旨）
- ESearch を Biopython ではなく **requests の POST + JSON** に切替（URL長と400を回避）
- PDAT 範囲はクエリ term に埋め込み（param の datetype/mindate/maxdate は不使用）
- 大量件数は PDAT 自動二分割でキャップ（既定 9,000 件/スライス）→ ID ページング
- 本文取得はしない（EFetch はアブストラクト等のメタ用に XML 取得のみ）
- 高密度日スライスのみ、必要に応じて **履歴方式(usehistory=y)へフォールバック**可

出力列（CSV）
title, doi(httpsリンク), pmid, journal, abstract,
taxon_labels（ラベルのみ）, taxon_ncbi_ids（空欄）, taxon_source,
pmcid, oa_url（空欄）, oa_license（空欄）, publisher_url（空欄）, publisher_license（空欄）,
mesh_ui（空欄）, mesh_tree_numbers（空欄）
"""
import csv
import os
import re
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from xml.etree import ElementTree as ET
from Bio import Entrez

# ---------------------------
# 語彙（ベース）
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
        reader = csv.reader(f)
        for row in reader:
            row = [c.strip() for c in row if c and c.strip()]
            if row:
                terms.append(row[0])
    return terms

# ---------------------------
# クエリ整形 & PDAT 範囲
# ---------------------------

def sanitize_term(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def range_term(base: str, mindate: str, maxdate: str) -> str:
    base_clean = sanitize_term(base)
    # スペース無しのコロンで安定化（400回避の経験則）
    return f"({base_clean}) AND ({mindate}[PDAT]:{maxdate}[PDAT])"

# ---------------------------
# E-utilities POST (JSON) ヘルパ
# ---------------------------

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def _ua():
    return "WholeBIF-Harvester/1.0 (+https://example.org; mailto:harvest@example.org)"

def _common_params():
    p = {}
    if getattr(Entrez, "api_key", None):
        p["api_key"] = Entrez.api_key
    if getattr(Entrez, "email", None):
        p["email"] = Entrez.email
    p["tool"] = "WholeBIF-Harvester"
    return p

def _post_json(endpoint: str, params: dict) -> dict:
    url = f"{EUTILS_BASE}/{endpoint}"
    allp = {**_common_params(), **params, "retmode": "json"}
    r = requests.post(url, data=allp, headers={"User-Agent": _ua()}, timeout=60)
    r.raise_for_status()
    return r.json()

# ---------------------------
# ESearch（POST+JSON）
# ---------------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def esearch_count(query: str, mindate: Optional[str] = None, maxdate: Optional[str] = None) -> int:
    term = range_term(query, mindate, maxdate) if (mindate and maxdate) else sanitize_term(query)
    j = _post_json("esearch.fcgi", {
        "db": "pubmed",
        "term": term,
        "retmax": 0,
        "usehistory": "n",
    })
    return int(j["esearchresult"]["count"])

@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def esearch_ids_window(query: str, retstart: int, retmax: int,
                       mindate: Optional[str] = None, maxdate: Optional[str] = None) -> List[str]:
    term = range_term(query, mindate, maxdate) if (mindate and maxdate) else sanitize_term(query)
    j = _post_json("esearch.fcgi", {
        "db": "pubmed",
        "term": term,
        "retstart": retstart,
        "retmax": retmax,
        "usehistory": "n",
    })
    return j["esearchresult"].get("idlist", [])

# 履歴（必要時フォールバック）
@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def esearch_history(term_or_query: str):
    j = _post_json("esearch.fcgi", {
        "db": "pubmed",
        "term": sanitize_term(term_or_query),
        "retmax": 0,
        "usehistory": "y",
    })
    res = j["esearchresult"]
    return int(res["count"]), res.get("webenv"), res.get("querykey")

# ESummary は Biopython（POST使用・XML→Python）で十分安定
@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def esummary_by_ids(id_list: List[str]):
    if not id_list:
        return []
    h = Entrez.esummary(db="pubmed", id=",".join(id_list), retmode="xml")
    records = Entrez.read(h); h.close()
    return records

@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def esummary_by_history(webenv: str, qk: str, retstart: int, retmax: int):
    h = Entrez.esummary(db="pubmed", webenv=webenv, query_key=qk, retstart=retstart, retmax=retmax, retmode="xml")
    records = Entrez.read(h); h.close()
    return records

@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 30), retry=retry_if_exception_type(Exception))
def efetch_pubmed_xml(pmids: List[str]) -> str:
    h = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml")
    data = h.read(); h.close()
    return data

# ---------------------------
# PubMed XML パース & Taxon（ラベルのみ）
# ---------------------------

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
        if pmid_el is None or not pmid_el.text:
            continue
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
                if kw.text:
                    keywords.append(kw.text.strip())

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

def detect_taxa_labels(mesh_desc_only: List[str], keywords: List[str], abstract: str) -> Tuple[List[str], str]:
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
        if ks:
            source = "Keyword"; labels = ks
        else:
            as_ = try_extract([abstract])
            if as_:
                source = "Abstract"; labels = as_
            else:
                return [], ""
    return labels, source

# ---------------------------
# CSV
# ---------------------------

DEFAULT_FIELDS = [
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
            return csv.DictReader(f).fieldnames
    except Exception:
        return None

def open_writer_with_compat(out_csv: str):
    existing = detect_existing_header(out_csv)
    if existing and len(existing) > 0:
        f = open(out_csv, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=existing)
        return f, writer, existing
    f = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=list(DEFAULT_FIELDS))
    writer.writeheader()
    return f, writer, list(DEFAULT_FIELDS)

# ---------------------------
# PDAT 分割
# ---------------------------

def dt_from_str(s: str) -> datetime:
    return datetime.strptime(s, "%Y/%m/%d")

def dt_to_str(d: datetime) -> str:
    return d.strftime("%Y/%m/%d")

def split_range(query: str, start: datetime, end: datetime, cap: int, sleep: float,
                allow_history_if_stuck: bool) -> List[Tuple[datetime, datetime, int, bool]]:
    if end < start:
        start, end = end, start
    mindate = dt_to_str(start); maxdate = dt_to_str(end)
    try:
        cnt = esearch_count(query, mindate, maxdate)
    except Exception:
        time.sleep(sleep)
        cnt = esearch_count(query, mindate, maxdate)

    if cnt <= cap:
        return [(start, end, cnt, False)]
    if (end - start).days <= 0:
        return [(start, end, cnt, True)]

    mid = start + (end - start) / 2
    left = split_range(query, start, mid, cap, sleep, allow_history_if_stuck)
    right = split_range(query, mid + timedelta(days=1), end, cap, sleep, allow_history_if_stuck)
    return left + right

# ---------------------------
# 1行書き出し
# ---------------------------

def write_row(writer, fieldnames, existing_pmids, pmid, rec, x):
    title = rec.get("Title") or ""
    if isinstance(title, list):
        title = title[0].get("content") if title else ""
    title = norm_space(str(title))

    journal = norm_space(rec.get("FullJournalName") or rec.get("Source") or "")

    doi = ""
    if "DOI" in rec:
        doi = (rec["DOI"] or "").strip()
    else:
        for item in rec.get("ArticleIds", []):
            if isinstance(item, dict) and item.get("Name", "").lower() == "doi":
                doi = (item.get("Value") or "").strip()
                if doi:
                    break
        if not doi and "ELocationID" in rec:
            val = rec["ELocationID"]
            if isinstance(val, list):
                for v in val:
                    if isinstance(v, dict) and (v.get("EIdType","").lower()=="doi"):
                        cand = (v.get("content") or "").strip()
                        if cand:
                            doi = cand; break
    doi_link = f"https://doi.org/{doi}" if doi else ""

    abstract = x.get("abstract","")
    mesh_desc_only = x.get("mesh_desc_only", [])
    keywords = x.get("keywords", [])
    pmcid = x.get("pmcid","")

    tax_labels, tax_source = detect_taxa_labels(mesh_desc_only, keywords, abstract)
    taxon_labels = "; ".join(tax_labels) if tax_labels else ""

    row = {
        "title": title,
        "doi": doi_link,
        "pmid": pmid,
        "journal": journal,
        "abstract": abstract,
        "taxon_labels": taxon_labels,
        "taxon_ncbi_ids": "",
        "taxon_source": tax_source,
        "pmcid": pmcid,
        "oa_url": "",
        "oa_license": "",
        "publisher_url": "",
        "publisher_license": "",
        "mesh_ui": "",
        "mesh_tree_numbers": "",
    }
    aligned = {k: row.get(k, "") for k in fieldnames}
    writer.writerow(aligned)
    existing_pmids.add(pmid)

# ---------------------------
# メイン
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--query", default=None)
    ap.add_argument("--tracers_csv", default=None)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=None)
    ap.add_argument("--max_count", type=int, default=None)
    ap.add_argument("--fast", action="store_true", help="外部照会はせず最速で収集")
    ap.add_argument("--cap_per_slice", type=int, default=9000)
    ap.add_argument("--date_start", default="1900/01/01")
    ap.add_argument("--date_end", default=None)
    ap.add_argument("--allow_history_if_stuck", action="store_true")
    args = ap.parse_args()

    # 既定（--fast は攻め）
    if args.fast:
        if args.chunk is None: args.chunk = 1000
        if args.sleep is None: args.sleep = 0.1
    else:
        if args.chunk is None: args.chunk = 300
        if args.sleep is None: args.sleep = 0.3

    # Entrez 設定（requests 側も _common_params() で活用）
    Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key

    # クエリ構築
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

    # 期間
    if not args.date_end:
        args.date_end = datetime.utcnow().strftime("%Y/%m/%d")
    start_dt = dt_from_str(args.date_start)
    end_dt = dt_from_str(args.date_end)

    # 再開用（既存PMID）
    existing_pmids = set()
    if os.path.exists(args.out_csv):
        with open(args.out_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pmid = (row.get("pmid") or "").strip()
                if pmid:
                    existing_pmids.add(pmid)

    f, writer, fieldnames = open_writer_with_compat(args.out_csv)

    try:
        # スライス作成
        slices = split_range(query, start_dt, end_dt, args.cap_per_slice, args.sleep, args.allow_history_if_stuck)
        total_est = sum(c for _,_,c,_ in slices)
        processed = 0
        pbar_total = args.max_count if args.max_count is not None else total_est

        with tqdm(total=pbar_total, desc="Harvesting", unit="rec") as pbar:
            for sdt, edt, cnt, need_hist in slices:
                mindate = dt_to_str(sdt); maxdate = dt_to_str(edt)

                if need_hist and args.allow_history_if_stuck:
                    # 高密度日 → 履歴方式
                    try:
                        subt, webenv, qk = esearch_history(range_term(query, mindate, maxdate))
                        retstart = 0
                        while retstart < subt:
                            if args.max_count is not None and processed >= args.max_count:
                                break
                            summaries = esummary_by_history(webenv, qk, retstart, args.chunk)
                            time.sleep(args.sleep)

                            page_pmids: List[str] = []
                            esum_by_pmid: Dict[str, dict] = {}
                            for rec in summaries:
                                pmid = rec.get("Id")
                                if not pmid or pmid in existing_pmids:
                                    continue
                                esum_by_pmid[pmid] = rec
                                page_pmids.append(pmid)

                            if page_pmids:
                                for i in range(0, len(page_pmids), args.chunk):
                                    sub = page_pmids[i:i+args.chunk]
                                    xml = efetch_pubmed_xml(sub)
                                    time.sleep(args.sleep)
                                    xml_info = parse_pubmed_xml(xml)
                                    for pmid in sub:
                                        if pmid not in xml_info:
                                            continue
                                        rec = esum_by_pmid.get(pmid, {})
                                        write_row(writer, fieldnames, existing_pmids, pmid, rec, xml_info.get(pmid, {}))
                                        f.flush()
                                        processed += 1; pbar.update(1)
                                        if args.max_count is not None and processed >= args.max_count:
                                            break
                            retstart += args.chunk
                    except Exception:
                        pass
                    continue

                # 通常スライス：ID ページング
                retstart = 0
                while retstart < cnt:
                    if args.max_count is not None and processed >= args.max_count:
                        break

                    id_list = esearch_ids_window(query, retstart, args.chunk, mindate, maxdate)
                    summaries = esummary_by_ids(id_list)
                    time.sleep(args.sleep)

                    page_pmids: List[str] = []
                    esum_by_pmid: Dict[str, dict] = {}
                    for rec in summaries:
                        pmid = rec.get("Id")
                        if not pmid or pmid in existing_pmids:
                            continue
                        esum_by_pmid[pmid] = rec
                        page_pmids.append(pmid)

                    if page_pmids:
                        for i in range(0, len(page_pmids), args.chunk):
                            sub = page_pmids[i:i+args.chunk]
                            xml = efetch_pubmed_xml(sub)
                            time.sleep(args.sleep)
                            xml_info = parse_pubmed_xml(xml)
                            for pmid in sub:
                                if pmid not in xml_info:
                                    continue
                                rec = esum_by_pmid.get(pmid, {})
                                write_row(writer, fieldnames, existing_pmids, pmid, rec, xml_info.get(pmid, {}))
                                f.flush()
                                processed += 1; pbar.update(1)
                                if args.max_count is not None and processed >= args.max_count:
                                    break

                    retstart += args.chunk

    finally:
        f.close()

if __name__ == "__main__":
    main()
