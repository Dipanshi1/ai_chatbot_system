"""
B1 Step 5 — offline knowledge-base indexing (spec §6.1.2).

Embeds the 160 cleaned training questions (ascending row_id) with the pinned MiniLM
revision and writes artifacts/kb_train_embeddings.npy (float32, 160 x 384, unit-norm rows).
Must run with HF_HUB_OFFLINE=1.

Usage (from the repo root):
  HF_HUB_OFFLINE=1 experiments/semantic_retrieval/.venv-b1/bin/python \
      experiments/semantic_retrieval/scripts/01_build_index.py
"""
import sys
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
for _p in (str(REPO_ROOT), str(EXPERIMENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import config  # noqa: E402
from lib import data as b1_data  # noqa: E402
from lib import provenance  # noqa: E402
from lib import retrievers  # noqa: E402


def main() -> int:
    split = b1_data.load_canonical_split()
    kb = split.train_df  # sorted by ascending row_id
    b1_data.assert_kb_is_train_only(kb.index, split)

    model = retrievers.load_embedding_model()
    t0 = time.perf_counter()
    index = retrievers.embed_texts(model, kb["User Message"].tolist())
    build_s = time.perf_counter() - t0
    retrievers.validate_index(index, len(kb))

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.INDEX_PATH, index)
    size = config.INDEX_PATH.stat().st_size
    print(f"Index written: {config.INDEX_PATH}")
    print(f"  shape={index.shape} dtype={index.dtype} rows=ascending train row_id")
    print(f"  build time (encode + normalise, batch_size={config.ENCODE_BATCH_SIZE}): {build_s:.3f} s")
    print(f"  file size: {size} bytes")
    print(f"  sha256: {provenance.sha256_file(config.INDEX_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
