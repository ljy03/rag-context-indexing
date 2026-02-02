#!/usr/bin/env python3
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INP = DATA_DIR / "mini_corpus_all_segments.jsonl"
OUT = DATA_DIR / "base_segments.jsonl"

def pick_first(obj, keys, default=None):
    for k in keys:
        v = obj.get(k)
        if v is not None:
            return v
    return default

def to_text(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "\n".join([str(x) for x in v if x is not None])
    return str(v)

written = 0
with INP.open("r", encoding="utf-8") as f, OUT.open("w", encoding="utf-8") as out:
    for line in f:
        obj = json.loads(line)

        seg_id = pick_first(obj, ["id", "segment_id", "docid"])
        seg_id = seg_id if isinstance(seg_id, str) else ""

        # segment text often stored under one of these:
        seg_text = pick_first(obj, ["contents", "content", "text", "segment", "passage", "body"])
        seg_text = to_text(seg_text).strip()

        # doc context candidates:
        title = to_text(pick_first(obj, ["title", "doc_title"])).strip()
        url = to_text(pick_first(obj, ["url", "doc_url"])).strip()

        headings = pick_first(obj, ["headings", "heading", "section_headings"], [])
        if isinstance(headings, str):
            headings = [headings]
        if not isinstance(headings, list):
            headings = []

        # derive doc_id from segment id
        doc_id = seg_id.split("#", 1)[0] if "#" in seg_id else pick_first(obj, ["doc_id", "document_id", "parent_docid"], "")
        doc_id = doc_id if isinstance(doc_id, str) else ""

        # skip if we can't identify an id or text
        if not seg_id or not seg_text:
            continue

        rec = {
            "id": seg_id,
            "doc_id": doc_id,
            "segment": seg_text,
            "title": title,
            "headings": headings,
            "url": url,
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1

print("Wrote records:", written)
print("Output:", OUT)
