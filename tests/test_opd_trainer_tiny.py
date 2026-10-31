"""End-to-end RP-OPD wiring with tiny random Llamas (CPU, no network): rollout -> teacher top-K ->
loss -> AdamW, HF export (stage-2 hand-off format), DCP resume, results layout, INDEX.csv."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from tests.tiny_models import make_model_dir, make_rows, make_tokenizer  # noqa: E402
from rrh.opd.batching import build_aligned_batch  # noqa: E402
from rrh.opd.loss import teacher_topk, topk_forward_kl  # noqa: E402
from rrh.results.layout import read_index  # noqa: E402
from rrh.utils.io import read_jsonl, write_jsonl  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    tok = make_tokenizer()
    s = make_model_dir(root / "student", 0, tok)
    t = make_model_dir(root / "teacher", 1, tok, hidden=48)
    data = root / "train.jsonl"
    write_jsonl(data, make_rows(12))
    return {"root": root, "student": s, "teacher": t, "data": data, "tok": tok}


def _args(tiny, tmp_path, steps):
    return ["--config", "configs/opd/base.yaml", "configs/opd/healthbench.yaml",
            "--run_id", "tiny__fake__hb__seed0__20261031-0000",
            "--results_root", str(tmp_path / "results"),
            "--set", f"student.path={tiny['student'].as_posix()}", f"teacher.path={tiny['teacher'].as_posix()}",
            "student.name=tiny", "student.attn_implementation=sdpa", "teacher.attn_implementation=sdpa",
            "student.gradient_checkpointing=false",
            f"data.train={tiny['data'].as_posix()}", "rollout.backend=hf", "rollout.max_response_tokens=6",
            "rollout.hf_batch_size=3", "train.global_batch=4", f"train.total_steps={steps}",
            "train.micro_batch_tokens=150", "train.save_hf_every=2", "train.save_hf_steps=[]",
            "train.save_resume_every=1", "optim.lr=1e-3", "loss.topk=16",
            f"paths.ckpt_root={(tmp_path / 'scratch').as_posix()}"]


def test_train_export_and_resume(tiny, tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM

    from rrh.opd import train_cli
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert train_cli.main(_args(tiny, tmp_path, 3)) == 0
    run = tmp_path / "results" / "rp_opd" / "tiny__fake__hb__seed0__20261031-0000"
    m = read_jsonl(run / "metrics" / "train.jsonl")
    assert [r["step"] for r in m] == [1, 2, 3]
    assert all(r["loss"] == r["loss"] and r["loss"] >= -1e-6 for r in m)        # finite, KL >= 0
    assert all(0 < r["teacher_topk_mass"] <= 1 + 1e-5 for r in m)
    assert all(r["micro_batches_mean"] >= 2 for r in m)                           # accumulation path
    ck = tmp_path / "scratch" / "rp_opd" / run.name
    for s in ("step_000000", "step_000002", "step_000003"):
        assert (ck / "hf" / s / "COMPLETE").exists(), s
    hf = ck / "hf" / "step_000003"
    model = AutoModelForCausalLM.from_pretrained(hf)                              # stage-2 loadable
    assert model.config.vocab_size == len(tiny["tok"])
    assert (hf / "tokenizer_config.json").exists() and (hf / "generation_config.json").exists()
    assert (ck / "resume" / "step_000003" / "trainer_state.json").exists()
    cfg_text = (run / "config.yaml").read_text()
    assert "git_commit" in cfg_text and "topk: 16" in cfg_text
    assert read_index(tmp_path / "results")[0]["status"] == "completed"

    # resume: same run id, more steps -> continues at step 4 with identical data order
    assert train_cli.main(_args(tiny, tmp_path, 5)) == 0
    m2 = read_jsonl(run / "metrics" / "train.jsonl")
    assert [r["step"] for r in m2] == [1, 2, 3, 4, 5]
    assert (ck / "hf" / "step_000005" / "COMPLETE").exists()
    w_resumed = AutoModelForCausalLM.from_pretrained(ck / "hf" / "step_000005").state_dict()
    w_before = model.state_dict()
    assert any(not torch.equal(w_resumed[k], w_before[k]) for k in w_before)     # it kept training


def test_distillation_reduces_kl_on_fixed_sequences(tiny):
    """Optimizing Eq. (4) on fixed student-context / teacher-context pairs drives the KL down."""
    from transformers import AutoModelForCausalLM

    from rrh.opd.contexts import build_context_ids
    tok = tiny["tok"]
    student = AutoModelForCausalLM.from_pretrained(tiny["student"], attn_implementation="eager")
    teacher = AutoModelForCausalLM.from_pretrained(tiny["teacher"], attn_implementation="eager").eval()
    rows = make_rows(4)
    ctx = [build_context_ids(tok, r) for r in rows]
    resp = [tok("The sky is blue.")["input_ids"] + [tok.eos_token_id] for _ in rows]
    sb = build_aligned_batch([c[0] for c in ctx], resp, tok.pad_token_id)
    tb = build_aligned_batch([c[1] for c in ctx], resp, tok.pad_token_id)
    with torch.no_grad():
        tl = teacher(input_ids=tb.input_ids, attention_mask=tb.attention_mask,
                     position_ids=tb.position_ids).logits[:, tb.logit_positions()]
    idx, logp, _ = teacher_topk(tl.reshape(-1, tl.shape[-1]), 16)
    mask = sb.resp_mask.reshape(-1)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-3)
    losses = []
    for _ in range(25):
        sl = student(input_ids=sb.input_ids, attention_mask=sb.attention_mask,
                     position_ids=sb.position_ids).logits[:, sb.logit_positions()]
        loss, _ = topk_forward_kl(sl.reshape(-1, sl.shape[-1]), idx, logp, mask)
        loss = loss / mask.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], losses
