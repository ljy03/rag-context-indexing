#!/usr/bin/env python3
"""
Encode corpus with Contriever (facebook/contriever) using mean pooling,
then build FAISS index. Used by 05_build_indices.sh for dense indexing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None].clamp(min=1e-9)
    return sentence_embeddings


def main():
    parser = argparse.ArgumentParser(description="Encode corpus with Contriever and build FAISS index")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus JSONL (id, contents)")
    parser.add_argument("--output", type=Path, required=True, help="Output index directory (e.g. indices/dense_standard/index)")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length per passage")
    parser.add_argument("--model", type=str, default="facebook/contriever", help="Contriever model name")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer and model: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    # Load corpus
    doc_ids = []
    texts = []
    with args.corpus.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            doc_ids.append(rec.get("id", ""))
            texts.append(rec.get("contents", ""))

    n = len(doc_ids)
    print(f"Encoding {n} documents with batch_size={args.batch_size}", file=sys.stderr)

    all_embeddings = []
    for i in range(0, n, args.batch_size):
        batch_texts = texts[i : i + args.batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        # L2-normalize for cosine similarity via dot product
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    # Build FAISS index (inner product = cosine for normalized vectors)
    try:
        import faiss
    except ImportError:
        print("faiss-cpu or faiss-gpu required: pip install faiss-cpu", file=sys.stderr)
        sys.exit(1)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Write index and docids (format expected by pyserini FaissSearcher)
    index_path = args.output / "index"
    faiss.write_index(index, str(index_path))

    docid_path = args.output / "docid"
    with docid_path.open("w", encoding="utf-8") as f:
        for doc_id in doc_ids:
            f.write(doc_id + "\n")

    print(f"Wrote FAISS index: {index_path}", file=sys.stderr)
    print(f"Wrote docids: {docid_path} ({n} docs)", file=sys.stderr)


if __name__ == "__main__":
    main()
