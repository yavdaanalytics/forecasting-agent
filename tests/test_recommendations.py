from forecasting_agent.recommendations.rank import confidence_label, rank_methods, recommend_segment


def test_rank_picks_lowest_wape():
    ranked = rank_methods({"prophet": 0.28, "ensemble": 0.32})
    assert ranked[0].method == "prophet"
    assert ranked[0].recommended is True
    assert ranked[1].method == "ensemble"


def test_confidence_bands():
    assert confidence_label(0.28) == "high"
    assert confidence_label(0.45) == "medium"
    assert confidence_label(0.58) == "low"
    assert confidence_label(None) == "unknown"


def test_recommend_segment_payload():
    rec = recommend_segment(
        "stable",
        {"prophet": 0.28, "ensemble": 0.35},
        horizon=60,
        num_skus=142,
    )
    assert rec["method"] == "prophet"
    assert rec["confidence"] == "high"
    assert rec["horizon_days"] == 60
    assert rec["num_skus"] == 142
    assert rec["selection_reason"] == "holdout_wape"


def test_recommend_empty_defaults_to_prophet():
    rec = recommend_segment("stable", {}, horizon=60, num_skus=0)
    assert rec["method"] == "prophet"
