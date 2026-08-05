import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.parquet"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


def main() -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    embeddings_path = (
        EMBEDDINGS_DIR / "all-MiniLM-L6-v2-text-embeddings.npy"
    )
    chunk_ids_path = (
        EMBEDDINGS_DIR / "all-MiniLM-L6-v2-text-chunk-ids.json"
    )

    chunks_df = pd.read_parquet(CHUNKS_PATH)

    baseline_df = chunks_df[
        ~chunks_df["is_sponsor_read"].fillna(False)
    ].copy()

    baseline_df = baseline_df[
        baseline_df["text"].fillna("").str.strip().ne("")
    ].copy()

    baseline_df = baseline_df[
        baseline_df["word_count"].fillna(0).ge(20)
    ].copy()

    texts = baseline_df["text"].astype(str).tolist()
    chunk_ids = baseline_df["chunk_id"].astype(str).tolist()

    print(f"Documents to embed: {len(texts)}")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    np.save(embeddings_path, embeddings)

    with chunk_ids_path.open("w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)

    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)
    print("Saved embeddings:", embeddings_path)
    print("Saved chunk IDs:", chunk_ids_path)


if __name__ == "__main__":
    main()