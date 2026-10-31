"""Hermetic tiny Llama student/teacher + char-level tokenizer with a Llama-3-style chat template."""

from __future__ import annotations

import string
from pathlib import Path

SPECIALS = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<pad>", "<unk>"]
TEMPLATE = ("{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>"
            "{{ m['content'] }}<|eot_id|>{% endfor %}"
            "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>{% endif %}")


def make_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    chars = sorted(set(string.printable))
    vocab = {t: i for i, t in enumerate(SPECIALS + chars)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tk.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<|begin_of_text|>",
                                  eos_token="<|eot_id|>", pad_token="<pad>", unk_token="<unk>")
    tok.add_special_tokens({"additional_special_tokens": SPECIALS[1:3]})
    tok.chat_template = TEMPLATE
    return tok


def make_model_dir(path: Path, seed: int, tok, hidden: int = 32, layers: int = 2) -> Path:
    import torch
    from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=4096, bos_token_id=tok.bos_token_id,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                      tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg)
    model.generation_config = GenerationConfig(eos_token_id=tok.eos_token_id,
                                               pad_token_id=tok.pad_token_id)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    return path


def make_rows(n: int = 12) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({"id": f"r{i}", "prompt": [{"role": "user", "content": f"Question {i}: why is the sky blue?"}],
                     "criteria": [f"Mentions scattering {i}.", "Avoids jargon."],
                     "weights": [5.0, -2.0] if i % 2 else [3.0, 1.0]})
    return rows
