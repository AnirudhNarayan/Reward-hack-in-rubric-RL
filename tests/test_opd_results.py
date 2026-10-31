"""Results layout, config resolution and INDEX.csv upserts."""

import yaml

from rrh.opd.config import load_opd_config
from rrh.results.layout import (RunDir, create_run, make_run_id, read_index, step_name,
                                update_index)


def test_run_id_format():
    rid = make_run_id("meta-llama/Llama-3.1-8B-Instruct", "llama-3.3-70b-judge", "healthbench", 0,
                      stamp="20261031-1200")
    assert rid == "llama-3.1-8b-instruct__llama-3.3-70b-judge__healthbench__seed0__20261031-1200"


def test_create_run_layout_and_config(tmp_path):
    rd = create_run(tmp_path / "results", "rp_opd", "llama-3.1-8b-instruct", "j", "hb", 1,
                    run_id="m__j__hb__seed1__20261031-1200")
    for d in ("logs", "metrics", "generations", "judgments", "figures", "tables"):
        assert (rd.path / d).is_dir()
    rd.write_config({"seed": 1, "optim": {"lr": 1e-5}})
    cfg = yaml.safe_load((rd.path / "config.yaml").read_text())
    assert cfg["optim"]["lr"] == 1e-5 and "git_commit" in cfg["_meta"]
    rd.write_config({"seed": 1})           # resumed segment does not overwrite the original
    assert yaml.safe_load((rd.path / "config.yaml").read_text())["optim"]["lr"] == 1e-5
    assert rd.generations("heldout", 20).as_posix().endswith("generations/heldout/step_000020.jsonl")
    assert rd.eval_metrics("heldout", "j", "teacher").name == "teacher.json"
    assert step_name(980) == "step_000980"


def test_checkpoint_link_points_to_scratch(tmp_path):
    rd = create_run(tmp_path / "results", "rp_opd", "m", "j", "d", 0, run_id="r")
    scratch = tmp_path / "scratch" / "r"
    ck = rd.link_checkpoints(scratch)
    assert ck.resolve() == scratch.resolve()
    assert RunDir(rd.path).checkpoint_dir().resolve() == scratch.resolve()


def test_index_upsert(tmp_path):
    root = tmp_path / "results"
    update_index(root, {"run_id": "a", "status": "running", "lr": 1e-5, "bogus": 1})
    update_index(root, {"run_id": "b", "status": "running"})
    update_index(root, {"run_id": "a", "status": "completed", "best_score": 0.5})
    rows = read_index(root)
    assert [r["run_id"] for r in rows] == ["a", "b"]
    assert rows[0]["status"] == "completed" and rows[0]["lr"] == "1e-05" and rows[0]["best_score"] == "0.5"
    assert "bogus" not in rows[0]


def test_config_layering_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/scratch/data")
    monkeypatch.delenv("CKPT_ROOT", raising=False)
    cfg = load_opd_config(["configs/opd/base.yaml", "configs/opd/rubrichub_science.yaml"],
                          ["train.total_steps=7", "rollout.backend=hf"])
    assert cfg["optim"]["lr"] == 3e-5 and cfg["rollout"]["max_response_tokens"] == 2048
    assert cfg["loss"]["topk"] == 256 and cfg["train"]["global_batch"] == 512
    assert cfg["train"]["total_steps"] == 7 and cfg["rollout"]["backend"] == "hf"
    assert cfg["data"]["train"] == "/scratch/data/opd/rubrichub_science/train.jsonl"
    assert cfg["paths"]["ckpt_root"] == "${CKPT_ROOT}"      # unresolved stays visible
    assert cfg["eval"]["temperature"] == 0.7 and cfg["eval"]["top_p"] == 0.8
