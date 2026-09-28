"""Extract the model architecture name from the raw snapshot's tags -> lower bound on substitutability for RQ1.

HF attaches config.json's model_type as a tag (e.g. llama, qwen2, gemma2). The same architecture does not imply the
same lineage (e.g. a model trained from scratch on the llama architecture), so grouping roots by this value
**underestimates** independent lineages = a lower bound on substitutability. Not used in the removal simulation.

Output: 02_data/processed/<snap>/arch.parquet  (model_id, arch)
Usage: python extract_arch.py ../../02_data/raw/hf_models_2026-09-25.jsonl.gz
"""
import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# tags used as architecture names (chosen from tag frequencies in a 20,000-model sample and the transformers/diffusers model_type lists)
ARCH = set("""
llama qwen qwen2 qwen2_moe qwen3 qwen3_moe qwen3_5 qwen3_next qwen2_vl qwen2_5_vl qwen2_audio qwen3_vl
mistral mixtral ministral gemma gemma2 gemma3 gemma3_text gemma3n paligemma recurrent_gemma
phi phi3 phi4 phimoe gpt2 gpt_neox gptj gpt_neo gpt_oss gpt_bigcode starcoder2 codegen falcon falcon_mamba bloom opt
stablelm olmo olmo2 olmoe cohere cohere2 deepseek_v2 deepseek_v3 exaone exaone4 internlm internlm2 chatglm baichuan
granite granitemoe jamba mamba mamba2 rwkv rwkv5 rwkv6 dbrx arctic minicpm minicpm3 smollm3 glm glm4 hunyuan
llava llava_next idefics2 idefics3 mllama florence2 blip blip-2 clip siglip siglip2 vit dinov2 deit beit swin convnext
resnet efficientnet mobilenet_v2 detr yolos rt_detr segformer mask2former sam
bert distilbert roberta xlm-roberta camembert deberta deberta-v2 electra albert mpnet modernbert xlnet longformer
t5 mt5 umt5 flan-t5 bart mbart marian pegasus led m2m_100 nllb
whisper wav2vec2 hubert wavlm speecht5 seamless_m4t vits bark musicgen encodec
stable-diffusion stable-diffusion-xl flux sd3 pixart kandinsky
""".split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    args = ap.parse_args()
    snap = args.raw.name.removeprefix("hf_models_").removesuffix(".jsonl.gz")
    ids, archs, seen = [], [], set()
    with gzip.open(args.raw, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            mid = r["id"]
            if mid in seen:
                continue
            seen.add(mid)
            hit = [t for t in (r.get("tags") or []) if t in ARCH]
            if hit:
                ids.append(mid)
                archs.append(hit[0])
    out = ROOT / "02_data" / "processed" / snap / "arch.parquet"
    d = pd.DataFrame({"model_id": ids, "arch": pd.Categorical(archs)})
    d.to_parquet(out, index=False)
    print(f"models with an architecture tag: {len(d):,} of {len(seen):,} ({100 * len(d) / len(seen):.1f}%)")
    print(d["arch"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
