"""Regex unit tests for the cleaning and inference rules.  python test_rules.py
The examples are names actually observed in the 2026-09-25 snapshot."""
import re

from clean_models import BOT_PATTERNS, COURSE_PATTERN, TEST_PATTERN
from infer_edges import QUANT_SUFFIX


def hit(pat, name):
    return re.search(pat, name.lower()) is not None


def test_bot_patterns():
    yes = {
        "timestamp_suffix": ["qwen-qwen1.5-0.5b-1725623935", "blockassist-bc-wild_loud_newt_1757349661"],
        "blockassist": ["blockassist-bc-pesty_extinct_prawn_1754946874", "blockassist"],
        "gensyn_swarm": ["qwen2.5-0.5b-instruct-gensyn-swarm-mangy_gentle_elk"],
        "uuid_name": ["53ae3e5c-8a9d-4454-af25-5033049d2df7"],
        "omega_miner": ["omega_hc944"],
        "cuid_name": ["cma5k9d0900tinegayi6a5ksk_cma5ke6kz00tpnegaicrpsprk"],
    }
    no = ["llama-3-8b-instruct", "gemma-2b-it", "bert-base-uncased", "candycarpetcleaningirving",
          "whisper-small-2024", "qwen2.5-7b-instruct-v1"]
    for k, names in yes.items():
        for nm in names:
            assert hit(BOT_PATTERNS[k], nm), (k, nm)
    for nm in no:
        assert not any(hit(p, nm) for p in BOT_PATTERNS.values()), nm


def test_course_and_test():
    assert hit(COURSE_PATTERN, "ppo-LunarLander-v2")
    assert hit(COURSE_PATTERN, "q-FrozenLake-v1-4x4-noSlippery")
    assert not hit(COURSE_PATTERN, "Qwen2.5-7B-Instruct-GRPO")
    for nm in ["test", "Test_2", "test.2", "my-model-tmp", "debug-run"]:
        assert hit(TEST_PATTERN, nm), nm
    for nm in ["attestation-model", "contest-winner", "latest-llama", "vit-bach-demo"]:
        assert not hit(TEST_PATTERN, nm), nm       # demo is left out on purpose


def test_quant_suffix():
    strip = lambda s: re.sub(QUANT_SUFFIX, "", s)
    assert strip("Llama-3.1-8B-Instruct-GGUF") == "Llama-3.1-8B-Instruct"
    assert strip("EXAONE-4.5-33B-i1-GGUF") == "EXAONE-4.5-33B"
    assert strip("Qwen2.5-7B-Instruct-AWQ") == "Qwen2.5-7B-Instruct"
    assert strip("Hermes-3-Llama-3.1-70B-4bit") == "Hermes-3-Llama-3.1-70B"
    assert strip("gemma-4-26B-uncensored-mlx-q8") == "gemma-4-26B-uncensored"
    assert strip("Llama-3.1-8B-Instruct") == "Llama-3.1-8B-Instruct"   # unchanged when there is no suffix


if __name__ == "__main__":
    test_bot_patterns(); test_course_and_test(); test_quant_suffix()
    print("all rule tests passed")
