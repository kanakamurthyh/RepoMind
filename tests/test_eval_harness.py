from __future__ import annotations

from evals.run_eval import evaluate


def test_hybrid_retrieval_beats_single_channel_baselines(shop_env):
    """Retrieval quality gate. The fixture is a tuning (dev) set, so thresholds guard regressions."""
    results = evaluate(*shop_env)
    hybrid = results["systems"]["repomind_hybrid"]

    assert hybrid["hit@5"] >= 0.95
    assert hybrid["mrr"] >= 0.9
    assert hybrid["mrr"] >= results["systems"]["dense_only"]["mrr"]
    assert hybrid["mrr"] >= results["systems"]["bm25_only"]["mrr"]
    assert hybrid["symbol_recall@5"] >= 0.85
