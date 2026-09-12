"""Complete a prompt using a locally trained checkpoint and its tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from gpt1.model import GPT, ModelConfig
from gpt1.train import device_for


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("gpt1/runs/wiki-small/latest.pt")
    )
    parser.add_argument("--data", type=Path, default=Path("gpt1/data/wiki-100m"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="0 selects greedy decoding"
    )
    parser.add_argument(
        "--top-k", type=int, default=40, help="0 samples the full vocabulary"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.temperature < 0 or args.top_k < 0 or args.max_new_tokens < 1:
        parser.error(
            "Temperature and top-k must be nonnegative; max-new-tokens must be positive"
        )
    manifest_bytes = (args.data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    # This checkpoint was created locally; never load untrusted pickle checkpoints.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if hashlib.sha256(manifest_bytes).hexdigest() != checkpoint["data_sha256"]:
        raise ValueError("Dataset manifest does not match checkpoint")
    tokenizer_path = args.data / "tokenizer.json"
    if (
        hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        != manifest["tokenizer_sha256"]
    ):
        raise ValueError("Tokenizer hash does not match training manifest")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prompt_ids = tokenizer.encode(args.prompt).ids
    if not prompt_ids:
        parser.error("Prompt must contain at least one token")
    config = ModelConfig(**checkpoint["settings"]["model"])
    if len(prompt_ids) > config.context:
        parser.error(f"Prompt exceeds the {config.context}-token context")
    device = device_for(args.device)
    model = GPT(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    torch.manual_seed(args.seed)
    ids = torch.tensor([prompt_ids], device=device)
    generated = []
    eos = tokenizer.token_to_id("<|endoftext|>")
    stop_reason = "max_new_tokens"
    for _ in range(args.max_new_tokens):
        logits = model(ids[:, -config.context :])[0][0, -1]
        if args.temperature == 0:
            next_id = int(logits.argmax().item())
        else:
            logits = logits / args.temperature
            if args.top_k:
                values, indices = logits.topk(min(args.top_k, config.vocab_size))
                choice = torch.multinomial(values.softmax(-1), 1)
                next_id = int(indices[choice].item())
            else:
                next_id = int(torch.multinomial(logits.softmax(-1), 1).item())
        generated.append(next_id)
        if next_id == eos:
            stop_reason = "eos"
            break
        ids = torch.cat((ids, torch.tensor([[next_id]], device=device)), dim=1)
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint["step"],
        "device": device,
        "prompt": args.prompt,
        "completion": tokenizer.decode(generated),
        "full_text": tokenizer.decode([*prompt_ids, *generated]),
        "generated_token_ids": generated,
        "stop_reason": stop_reason,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
