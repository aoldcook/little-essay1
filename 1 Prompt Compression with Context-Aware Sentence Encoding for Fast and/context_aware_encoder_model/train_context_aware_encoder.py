from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from context_aware_encoder_model.context_aware_sentence_encoder import (
    ContextAwareEncoderConfig,
    ContextAwareSentenceEncoder,
    build_marked_context_from_text,
)


class CQRDataset(Dataset):
    """
    JSONL 格式示例：
    {
      "question": "CPC为什么更稳？",
      "context": "......",
      "positive_sentence": "......",
      "negative_sentences": ["......", "......"]
    }
    """
    def __init__(self, path: Path, marker_start: str, marker_end: str):
        self.rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pos_marked = build_marked_context_from_text(
                    row["context"], row["positive_sentence"], marker_start, marker_end
                )
                neg_marked = [
                    build_marked_context_from_text(row["context"], s, marker_start, marker_end)
                    for s in row.get("negative_sentences", [])
                ]
                self.rows.append(
                    {
                        "question": row["question"],
                        "positive_marked_context": pos_marked,
                        "negative_marked_contexts": neg_marked,
                    }
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, List[str]]:
    return {
        "questions": [x["question"] for x in batch],
        "positive_marked_contexts": [x["positive_marked_context"] for x in batch],
        "negative_marked_contexts": [x["negative_marked_contexts"] for x in batch],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.05)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ContextAwareEncoderConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        temperature=args.temperature,
        device=device,
    )
    model = ContextAwareSentenceEncoder(config)

    dataset = CQRDataset(Path(args.train_file), config.marker_start, config.marker_end)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            optimizer.zero_grad()
            loss = model.contrastive_loss(
                questions=batch["questions"],
                positive_marked_contexts=batch["positive_marked_contexts"],
                negative_marked_contexts=batch["negative_marked_contexts"],
            )
            loss.backward()
            optimizer.step()

            bs = len(batch["questions"])
            total_loss += float(loss.item()) * bs
            total_count += bs

        print(f"epoch={epoch:02d} loss={total_loss / max(total_count, 1):.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(output_dir)
    model.tokenizer.save_pretrained(output_dir)
    with (output_dir / "encoder_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(config), f, ensure_ascii=False, indent=2)
    print("saved to", output_dir)


if __name__ == "__main__":
    main()
