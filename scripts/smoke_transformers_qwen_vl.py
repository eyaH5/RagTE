"""Smoke-test Qwen2.5-VL directly through Transformers."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(args.model)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Return exactly this JSON and nothing else: {\"answer\":\"hello\"}",
                }
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt").to(model.device)

    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated = generated[:, inputs.input_ids.shape[1] :]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
