"""第11章: 足し算 200 件の SFT データを生成."""
from __future__ import annotations
import json
import pathlib
import random


def main() -> None:
    random.seed(0)
    rows = []
    for _ in range(200):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        rows.append({
            "messages": [
                {"role": "user", "content": f"{a} + {b} はいくつですか？"},
                {"role": "assistant",
                 "content": f"<think>{a}と{b}を足すと{a+b}。</think>答えは{a+b}です。"},
            ]
        })
    pathlib.Path("tiny_addition.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )
    print("Wrote tiny_addition.jsonl with", len(rows), "rows")


if __name__ == "__main__":
    main()
