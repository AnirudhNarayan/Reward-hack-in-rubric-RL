"""Pre-download model weights into $HF_HOME (gated Llama/Gemma need HF_TOKEN in ~/.env).

    python -m rrh.data.fetch_models meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-70B-Instruct
"""

from __future__ import annotations

import sys

PATTERNS = ["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt", "*.tiktoken", "*.jinja"]
# duplicate reference-format weights (gpt-oss ships a second ~65 GB copy under original/)
IGNORE = ["original/*", "metal/*"]


def main(argv=None):
    from huggingface_hub import snapshot_download

    from rrh.utils.io import load_env
    load_env()                      # HF_TOKEN; never printed
    for m in (argv if argv is not None else sys.argv[1:]):
        p = snapshot_download(m, allow_patterns=PATTERNS, ignore_patterns=IGNORE)
        print(f"ok {m} -> {p}", flush=True)


if __name__ == "__main__":
    main()
