#!/usr/bin/env python3
"""
Encode corpus with Contriever (facebook/contriever): mean pooling + L2 normalize,
then build FAISS index. Used by build_indices.sh for dense indexing.
Query encoder must use the same: tokenize (same max_length), mean_pooling with attention_mask, L2 normalize.

Colab / fast encoding (e.g. 50k passages on T4):
  --batch-size 128 --max-length 256 --device cuda
  (Uses fp16 autocast on CUDA. If OOM, try --batch-size 64. max_length 256 is ~2× faster than 512.)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average over non-padding tokens. attention_mask: 1 = real token, 0 = pad. Returns (batch, dim)."""
    mask = attention_mask.unsqueeze(-1).float()
    masked = token_embeddings * mask
    sum_emb = masked.sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
    return sum_emb / sum_mask


def _encode_texts(tokenizer, model, texts, device, max_length=512):
    """Encode texts with Contriever: mean pooling + L2 normalize. Return L2-normalized (n, dim) float32."""
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    use_amp = (device == "cuda") if isinstance(device, str) else (getattr(device, "type", None) == "cuda")
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
    emb = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32)


class ContrieverQueryEncoder:
    """
    Query encoder that matches passage encoding: same tokenization (max_length),
    mean pooling with attention_mask, L2 normalize. Same space as index for correct cosine/IP scores.
    """

    def __init__(self, model_name: str = "facebook/contriever", device: str = None, max_length: int = 512):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()) else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, queries, batch_size: int = 64):
        """Encode query/queries; returns (dim,) for single query or (n, dim) for multiple. L2-normalized float32.
        Single-query 1D return is required by Pyserini FaissSearcher (len(emb_q) == dimension)."""
        if isinstance(queries, str):
            queries = [queries]
        all_emb = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            emb = _encode_texts(self.tokenizer, self.model, batch, self.device, self.max_length)
            all_emb.append(emb)
        out = np.vstack(all_emb)
        if out.shape[0] == 1:
            return out[0]  # (768,) so Pyserini's len(emb_q) == self.dimension
        return out


def main():
    parser = argparse.ArgumentParser(description="Encode corpus with Contriever and build FAISS index")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus JSONL (id, contents)")
    parser.add_argument("--output", type=Path, required=True, help="Output index directory (e.g. indices/dense_standard/index)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (use 128 on T4; 64 if OOM)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length (256 is ~2× faster, often enough for retrieval)")
    parser.add_argument("--model", type=str, default="facebook/contriever", help="Contriever model name")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer and model: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    # Load corpus (strict docid checks to keep embedding order aligned with docids)
    doc_ids = []
    texts = []
    with args.corpus.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            doc_id = rec.get("id")
            assert doc_id is not None and doc_id != "", (
                "Corpus must have non-empty 'id' for every record; got id=%r" % (doc_id,)
            )
            doc_ids.append(doc_id)
            texts.append(rec.get("contents", ""))

    assert len(set(doc_ids)) == len(doc_ids), (
        "Corpus has duplicate doc ids; docid–embedding alignment would be wrong"
    )
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
            if args.device == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
        embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
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
