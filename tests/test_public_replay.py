import importlib.util
import json
from pathlib import Path

import pytest
from sentinelrag.utils.stats import bernoulli_mixture_e_value

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_reproduce", ROOT / "reproduce.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("p0", [.15, .3004170799667139, .2883755391484502, .3906278341482568])
def test_replay_matches_production_statistics(p0):
    for n in range(1, 31):
        for s in range(n+1):
            assert module.e_value(s,n,p0) == pytest.approx(bernoulli_mixture_e_value(s,n,p0), rel=1e-11)


def test_all_archived_trajectories_reproduce_conditions():
    payload = json.loads((ROOT / "results/trajectories.json").read_text())
    actual = module.condition_rows(payload["traces"])
    assert actual == json.loads((ROOT / "results/conditions.json").read_text())
    positive = [r for r in actual if r["study"] == "primary" and r["state"] == "watermarked"]
    assert len(positive) == 288
    for method, success, cost in (("SentinelRAG", .7476851851851851, 8.30787037037037), ("RAVET", 5/6, 6.659722222222222)):
        m = module.summarize([r for r in positive if r["method"] == method],15)
        assert m["vsr"] == pytest.approx(success)
        assert m["mq"] == pytest.approx(cost)


def test_repeated_identity_rejected():
    with pytest.raises(ValueError, match="distinct evidence"):
        module.replay({"labels":[True,True],"unit_order":["x","x"],"p0":.15,"alpha":.01})


def test_budget_beyond_archive_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        module.summarize([{"curve":[0,0]}],3)
