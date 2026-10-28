"""
Resource measurement (spec §11.1). Values are measured, never assumed.

Two parts:
  * in-process helpers: environment record, RSS checkpoints, latency harness;
  * a fresh-process probe (`python lib/resources.py --probe ...`) that takes RSS0 after
    importing numpy/pandas/scikit-learn and Stage A modules but BEFORE importing torch /
    sentence-transformers, then records the import/model/index/after-retrieval checkpoints,
    model load time, index build/load time, and peak RSS (ru_maxrss).

Incremental peak RSS = peak RSS − RSS0 (the gated value). The probe also records an extra
checkpoint after loading Stage A data (still before torch); it is included in the
incremental figure, which makes the gate conservative.
"""
import os
import platform
import resource
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import psutil

MIB = 1024 * 1024


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / MIB


def peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _cpu_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def environment_record() -> Dict[str, object]:
    uname = platform.uname()
    vm = psutil.virtual_memory()
    record: Dict[str, object] = {
        "os": uname.system,
        "kernel": uname.release,
        "wsl": "microsoft" in uname.release.lower(),
        "cpu_model": _cpu_model(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_total_visible_mb": round(vm.total / MIB, 1),
        "ram_available_mb_at_start": round(vm.available / MIB, 1),
        "python_version": platform.python_version(),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    }
    if "torch" in sys.modules:
        record["torch_threads"] = sys.modules["torch"].get_num_threads()
    return record


class Checkpoints:
    def __init__(self) -> None:
        self.values: Dict[str, float] = {}

    def mark(self, name: str) -> float:
        self.values[name] = round(rss_mb(), 2)
        return self.values[name]


def latency_stats(samples_ms: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "max_ms": float(arr.max()),
    }


def measure_latency(retriever, queries: Sequence[str], calls: int, k: int) -> Dict[str, object]:
    """
    Cold first query (timed separately), then `calls` single-query calls cycling through
    `queries` in the given order. End-to-end = encode (clean_text + representation) +
    search (scores + top-k); search-only = search part.
    """
    t0 = time.perf_counter()
    retriever.rank(queries[0], k)
    cold_ms = (time.perf_counter() - t0) * 1000
    e2e: List[float] = []
    search_only: List[float] = []
    for i in range(calls):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        rep = retriever.encode(q)
        t1 = time.perf_counter()
        retriever.search(rep, k)
        t2 = time.perf_counter()
        e2e.append((t2 - t0) * 1000)
        search_only.append((t2 - t1) * 1000)
    return {
        "cold_first_query_ms": cold_ms,
        "end_to_end": latency_stats(e2e),
        "search_only": latency_stats(search_only),
        "batch_size": 1,
        "k": k,
    }


# ---------------------------------------------------------------------------
# Fresh-process probe
# ---------------------------------------------------------------------------
def _probe_main(argv: Sequence[str]) -> None:
    import argparse
    import json
    from pathlib import Path

    lib_dir = Path(__file__).resolve().parent
    here = lib_dir.parent  # experiments/semantic_retrieval
    repo = here.parent.parent
    # Running this file as a script puts lib/ first on sys.path, where module names such as
    # data/stats/metrics could shadow third-party modules. Remove it.
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != lib_dir]
    for p in (str(repo), str(here)):
        if p not in sys.path:
            sys.path.insert(0, p)

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", required=True)
    parser.add_argument("--queries-json", required=True,
                        help="JSON list of {query, target_intent}")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    # Imports that count towards the baseline (no torch / sentence-transformers yet).
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
    import config
    from lib import data as b1_data
    from lib import retrievers as b1_retrievers
    from src.ml.inference import IntentClassifier
    from src.ml.preprocessing import clean_text

    if "torch" in sys.modules or "sentence_transformers" in sys.modules:
        raise RuntimeError("torch was imported before RSS0; probe invalid")

    cp = Checkpoints()
    cp.mark("rss0_baseline_process")

    items = json.loads(Path(args.queries_json).read_text())
    split = b1_data.load_canonical_split()
    classifier = IntentClassifier()
    predicted = [classifier.predict(clean_text(it["query"])) for it in items]
    cp.mark("rss_stage_a_data_loaded")

    import torch  # noqa: F401
    import sentence_transformers  # noqa: F401
    cp.mark("rss_import")

    t0 = time.perf_counter()
    model = b1_retrievers.load_embedding_model()
    model_load_s = time.perf_counter() - t0
    cp.mark("rss_model")

    questions = split.train_df["User Message"].tolist()
    t0 = time.perf_counter()
    built = b1_retrievers.embed_texts(model, questions)
    index_build_s = time.perf_counter() - t0
    del built

    t0 = time.perf_counter()
    index = np.load(config.INDEX_PATH)
    index_load_s = time.perf_counter() - t0
    retriever = b1_retrievers.SemanticRetriever(split.train_df, model=model, index=index)
    cp.mark("rss_index")

    for it, pred in zip(items, predicted):
        retriever.rank(it["query"], config.RETRIEVAL_DEPTH)                                 # Pass 1
        retriever.rank(it["query"], config.RETRIEVAL_DEPTH, intent=it.get("target_intent"))  # Pass 2
        retriever.rank(it["query"], config.RETRIEVAL_DEPTH,
                       intent=pred.predicted_intent if pred else None)                     # Pass 3
    cp.mark("rss_after_retrieval")

    peak = round(peak_rss_mb(), 2)
    out = {
        "checkpoints_mb": cp.values,
        "peak_rss_mb": peak,
        "incremental_peak_rss_mb": round(peak - cp.values["rss0_baseline_process"], 2),
        "model_load_s": model_load_s,
        "index_build_s": index_build_s,
        "index_build_batch_size": config.ENCODE_BATCH_SIZE,
        "index_load_s": index_load_s,
        "n_queries": len(items),
        "environment": environment_record(),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))


def run_probe(queries: Sequence[Dict[str, Optional[str]]], workdir: str, python: str = sys.executable) -> Dict[str, object]:
    """Run one fresh-process probe and return its JSON result."""
    import json
    import subprocess
    from pathlib import Path

    qpath = Path(workdir) / "probe_queries.json"
    opath = Path(workdir) / "probe_out.json"
    qpath.write_text(json.dumps(list(queries)))
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [python, str(Path(__file__).resolve()), "--probe", "--queries-json", str(qpath), "--out", str(opath)],
        check=True, env=env, capture_output=True, text=True,
    )
    return json.loads(opath.read_text())


if __name__ == "__main__":
    _probe_main(sys.argv[1:])
