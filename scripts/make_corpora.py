#!/usr/bin/env python3
"""
Create standard and contextualized corpus variants from MS MARCO V2.1 segment format.
Reads mini_corpus (docid, segment, title, headings, url) directly; no normalization step.
Contextualized: segment first, then raw title only (no [Context: ...] wrapper).
"""
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INP = DATA_DIR / "mini_corpus_all_segments.jsonl"
OUT_STD = DATA_DIR / "corpus_standard.jsonl"
OUT_CTX = DATA_DIR / "corpus_contextualized.jsonl"

if not INP.exists():
    print(f"Error: input not found: {INP}", file=sys.stderr)
    print("Run extract_segments.py first.", file=sys.stderr)
    sys.exit(1)


def get_title(doc_meta):
    """Title only (short), no wrapper."""
    title = (doc_meta.get("title") or "").strip()
    return title


def to_text(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "\n".join([str(x) for x in v if x is not None])
    return str(v)


# Official MS MARCO V2.1 segment fields: docid, url, title, headings, segment, start_char, end_char
records = []
doc_map = {}
with INP.open("r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        seg_id = (obj.get("docid") or "").strip()
        seg_text = to_text(obj.get("segment", "")).strip()
        if not seg_id or not seg_text:
            continue
        doc_id = seg_id.split("#", 1)[0] if "#" in seg_id else seg_id
        title = to_text(obj.get("title", "")).strip()
        url = to_text(obj.get("url", "")).strip()
        headings = obj.get("headings")
        if isinstance(headings, str):
            headings = [headings]
        if not isinstance(headings, list):
            headings = []

        rec = {
            "id": seg_id,
            "doc_id": doc_id,
            "segment": seg_text,
            "title": title,
            "headings": headings,
            "url": url,
        }
        records.append(rec)
        if doc_id not in doc_map:
            doc_map[doc_id] = {"title": title, "headings": headings, "url": url}

n = 0
with OUT_STD.open("w", encoding="utf-8") as out_std, \
     OUT_CTX.open("w", encoding="utf-8") as out_ctx:
    for rec in records:
        seg = rec["segment"]
        doc_id = rec.get("doc_id", "")
        doc_meta = doc_map.get(doc_id, {})

        # Standard: just segment text
        out_std.write(json.dumps({"id": rec["id"], "contents": seg}, ensure_ascii=False) + "\n")

        # Contextualized: segment then raw title (no wrapper tokens)
        title = get_title(doc_meta)
        if title:
            contents_ctx = f"{seg}\n\n{title}"
        else:
            contents_ctx = seg
        out_ctx.write(json.dumps({"id": rec["id"], "contents": contents_ctx}, ensure_ascii=False) + "\n")

        n += 1

print("Segments processed:", n)
print("Wrote:", OUT_STD)
print("Wrote:", OUT_CTX)
