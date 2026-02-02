#!/usr/bin/env python3
"""
Create standard and contextualized corpus variants.
Contextualized: segment FIRST, then light context (title/headings).
"""
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INP = DATA_DIR / "base_segments.jsonl"
OUT_STD = DATA_DIR / "corpus_standard.jsonl"
OUT_CTX = DATA_DIR / "corpus_contextualized.jsonl"

def build_context_suffix(doc_meta):
    """Build a SHORT context suffix (title + headings only)."""
    parts = []
    if doc_meta.get("title"):
        parts.append(f"Title: {doc_meta['title']}")
    hs = doc_meta.get("headings") or []
    if hs:
        parts.append("Section: " + " > ".join(hs[:3]))
    if parts:
        return " | ".join(parts)
    return ""

records = []
doc_map = {}
with INP.open("r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        seg = rec["segment"].strip()
        rec["segment"] = seg
        records.append(rec)

        doc_id = rec.get("doc_id", "")
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "title": rec.get("title", ""),
                "headings": rec.get("headings") or [],
                "url": rec.get("url", ""),
            }

n = 0
with OUT_STD.open("w", encoding="utf-8") as out_std, \
     OUT_CTX.open("w", encoding="utf-8") as out_ctx:
    for rec in records:
        seg = rec["segment"]
        doc_id = rec.get("doc_id", "")
        doc_meta = doc_map.get(doc_id, {})

        # Standard: just segment text
        out_std.write(json.dumps({"id": rec["id"], "contents": seg}, ensure_ascii=False) + "\n")

        # Contextualized: segment FIRST, then context
        context = build_context_suffix(doc_meta)
        if context:
            contents_ctx = f"{seg}\n\n[Context: {context}]"
        else:
            contents_ctx = seg
        out_ctx.write(json.dumps({"id": rec["id"], "contents": contents_ctx}, ensure_ascii=False) + "\n")

        n += 1

print("Segments processed:", n)
print("Wrote:", OUT_STD)
print("Wrote:", OUT_CTX)
