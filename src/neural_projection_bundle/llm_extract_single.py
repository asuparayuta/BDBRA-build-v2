
import os, json, argparse, csv
from typing import Dict, Any, List
from openai import OpenAI
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from io import BytesIO
from prompts_llm import SYSTEM_PROMPT, USER_TEMPLATE, JSON_SCHEMA

def read_pdf_text(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    laparams = LAParams(line_margin=0.2, word_margin=0.1, char_margin=2.0)
    return extract_text(BytesIO(data), laparams=laparams) or ""

def call_openai_structured(text: str, meta: Dict[str, str], region_hints: str, model: str) -> List[Dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    user_msg = USER_TEMPLATE.format(
        title=meta.get("title",""), journal=meta.get("journal",""),
        year=meta.get("year",""), doi=meta.get("doi",""),
        pmid=meta.get("pmid",""), pmcid=meta.get("pmcid",""),
        region_hints=region_hints, text=text[:18000]
    )
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, temperature=0.0,
        response_format={"type":"json_schema","json_schema":JSON_SCHEMA},
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_msg}]
    )
    data = json.loads(resp.choices[0].message.content)
    return data.get("items", [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf")
    ap.add_argument("--text")
    ap.add_argument("--out_csv", default="out_single.csv")
    ap.add_argument("--model", default="gpt-5-nano")
    ap.add_argument("--title", default="")
    ap.add_argument("--journal", default="")
    ap.add_argument("--year", default="")
    ap.add_argument("--doi", default="")
    ap.add_argument("--pmid", default="")
    ap.add_argument("--pmcid", default="")
    ap.add_argument("--taxon", default="")
    ap.add_argument("--method", default="")
    ap.add_argument("--reference", default="")
    ap.add_argument("--region_hints", default="M1,V1,CPu,MD,CA1,Thalamus,Cerebellum,STN,SNc,GPe,GPi,PPN,SMA,pre-SMA,PFC,Hippocampus,Putamen,Caudate")
    args = ap.parse_args()

    if not args.text and not args.pdf:
        raise SystemExit("Provide --pdf or --text")

    text = args.text if args.text else read_pdf_text(args.pdf)
    meta = {
        "title": args.title, "journal": args.journal, "year": args.year, "doi": args.doi,
        "pmid": args.pmid, "pmcid": args.pmcid
    }
    items = call_openai_structured(text, meta, args.region_hints, args.model)

    header = ["sender","receiver","reference","journal","DOI","Taxon","Method","Pointer","Figure"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader()
        for it in items:
            w.writerow({
                "sender": it.get("sender",""),
                "receiver": it.get("receiver",""),
                "reference": args.reference or args.title,
                "journal": args.journal,
                "DOI": args.doi,
                "Taxon": args.taxon or it.get("taxon_hint","Unspecified"),
                "Method": args.method or it.get("method_hint","Unspecified"),
                "Pointer": it.get("pointer",""),
                "Figure": ", ".join(it.get("figure_ids",[]) or ([it.get("figure","")] if it.get("figure") else []))
            })
    print(f"Wrote CSV to {args.out_csv}")

if __name__ == "__main__":
    main()
