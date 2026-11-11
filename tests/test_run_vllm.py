"""The GPU runner (scripts/run_vllm.py) must write the same records as run_mlx.py.

Generation is mocked: a fake tokenizer turns token ids into text, and a fake
`vllm` module stands in for the real one, so these tests run without a GPU and
without vLLM installed.
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_vllm  # noqa: E402

from cot_faithfulness.inference.mlx_records import build_record  # noqa: E402

EOS = 99
VOCAB = {1: "<think>\n", 2: "The professor says (A).", 3: "\n</think>\n\n", 4: "Answer: (B)", 5: " hmm", EOS: "<|im_end|>"}


def decode(ids):
    return "".join(VOCAB[i] for i in ids)


def fake_job():
    ex = SimpleNamespace(task="fake_task", example_id="fake_task_0", correct_letter="A")
    var = SimpleNamespace(variant_id="sycophancy_B", hint_type="sycophancy", prompt="FAKE prompt",
                          display_options={"A": "Yes", "B": "No"}, letter_map={"A": "A", "B": "B"},
                          hint_suggested_letter="B", hint_description="FAKE hint")
    return ex, var


def mlx_record(ex, var, streamed, finish_reason, opens_think, k, decoding):
    """What run_mlx.py does with the tokens it streams: it never keeps the end token."""
    tokens = [t for t in streamed if not (finish_reason == "stop" and t == EOS)]
    return build_record(model_id="FAKE", example=ex, variant=var, cot=decode(tokens).strip(),
                        n_new_tokens=len(tokens), finish_reason=finish_reason, prompt_opens_think=opens_think,
                        extra={"sample_index": k, "decoding": decoding})


def check_same(token_ids, finish_reason, opens_think):
    ex, var = fake_job()
    gpu_decoding = {"backend": "vllm", "dtype": "bfloat16", "temperature": 0.7}
    gpu = run_vllm.make_record(model_id="FAKE", example=ex, variant=var, token_ids=token_ids,
                               finish_reason=finish_reason, stop_ids=[EOS], decode=decode,
                               opens_think=opens_think, sample_index=3, decoding=gpu_decoding)
    mlx = mlx_record(ex, var, token_ids, finish_reason, opens_think, 3, {"temperature": 0.7})
    assert list(gpu) == list(mlx)  # same fields in the same order
    assert gpu["backend"] == "vllm" and mlx["backend"] == "mlx"
    assert gpu["decoding"] == gpu_decoding
    for key in mlx:
        if key not in ("backend", "decoding"):
            assert gpu[key] == mlx[key], key
    return gpu


def test_instruct_trace_matches_mlx():
    rec = check_same([5, 4, EOS], "stop", opens_think=False)  # vLLM returns the end token, MLX does not
    assert rec["cot"] == "hmmAnswer: (B)" and rec["n_new_tokens"] == 2 and rec["parsed_answer"] == "B"


def test_thinking_trace_matches_mlx():
    rec = check_same([1, 2, 3, 4, EOS], "stop", opens_think=False)
    assert rec["parsed_answer"] == "B" and rec["answer_source"] == "answer_text"
    assert rec["thinking"] == "The professor says (A)."


def test_template_opened_thinking_cut_by_length_matches_mlx():
    # R1-distill shape: the prompt ends in "<think>", the output never closes it.
    rec = check_same([2, 5, 5], "length", opens_think=True)
    assert rec["thinking_closed"] is False and rec["n_new_tokens"] == 3


def test_vllm_generate_glue(monkeypatch):
    """load_vllm passes our settings to vLLM and returns samples in index order."""
    seen = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            seen["llm"] = kwargs

        def generate(self, prompts, params, use_tqdm=False):
            seen["prompts"], seen["params"] = prompts, params
            outs = [SimpleNamespace(index=1, token_ids=(4, EOS), finish_reason="stop"),
                    SimpleNamespace(index=0, token_ids=(5, 5), finish_reason="length")]
            return [SimpleNamespace(outputs=outs) for _ in prompts]

    fake = ModuleType("vllm")
    fake.LLM, fake.SamplingParams = FakeLLM, lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", fake)
    args = SimpleNamespace(model="FAKE", dtype="bfloat16", tensor_parallel_size=1, gpu_memory_utilization=0.9,
                           seed=0, temperature=0.7, top_p=0.95, n_samples=2, max_tokens=4096)
    generate = run_vllm.load_vllm(args, max_model_len=5000, stop_ids=[EOS])
    result = generate([[7, 8, 9]], [1234])
    assert result == [[([5, 5], "length"), ([4, EOS], "stop")]]
    assert seen["prompts"] == [{"prompt_token_ids": [7, 8, 9]}]  # token ids, so vLLM adds no BOS
    p = seen["params"][0]
    assert (p["n"], p["temperature"], p["top_p"], p["max_tokens"], p["seed"]) == (2, 0.7, 0.95, 4096, 1234)
    assert p["stop_token_ids"] == [EOS]
    assert seen["llm"]["generation_config"] == "vllm" and seen["llm"]["dtype"] == "bfloat16"


def test_prompt_seeds_are_fixed_and_distinct():
    a = run_vllm.prompt_seed(0, "causal_judgement_3", "baseline")
    assert a == run_vllm.prompt_seed(0, "causal_judgement_3", "baseline")
    assert a != run_vllm.prompt_seed(0, "causal_judgement_3", "placebo")
    assert a != run_vllm.prompt_seed(1, "causal_judgement_3", "baseline")


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["run_vllm.py", "--backend", "hf", "--model", "org/FAKE", *argv])
    run_vllm.main()


def test_refuses_to_resume_with_other_seed_or_sample_count(tmp_path, monkeypatch):
    import json

    import pytest

    out = tmp_path / "org__FAKE"
    out.mkdir()
    base = {"backend": "hf", "dtype": "bfloat16", "temperature": 0.7, "top_p": 1.0, "max_tokens": 768,
            "chat_template_kwargs": {}, "seed": 0, "n_samples": 5}
    for change in ({"seed": 1}, {"n_samples": 3}, {"top_p": 0.95}):
        json.dump({**base, **change}, open(out / "run_config.json", "w"))
        with pytest.raises(SystemExit, match="other settings"):
            run_main(monkeypatch, ["--temperature", "0.7", "--n-samples", "5", "--max-tokens", "768",
                                   "--out-root", str(tmp_path)])


def test_refuses_the_released_greedy_folder(monkeypatch):
    import pytest

    with pytest.raises(SystemExit):
        run_main(monkeypatch, ["--out-root", str(run_vllm.GREEDY_ROOT)])
