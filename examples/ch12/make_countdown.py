"""第12章: Countdown 問題の生成."""
from __future__ import annotations
import json
import random


def gen_problem(rng: random.Random) -> dict:
    while True:
        nums = rng.sample(range(1, 20), 4)
        a, b, c, d = nums
        ops = [
            (a + b) * (c - d) if c != d else None,
            (a * b) - (c + d),
            (a + b) + (c - d),
            (a - b) * (c + d) if a != b else None,
        ]
        candidates = [abs(v) for v in ops if v is not None and abs(v) > 0]
        if candidates:
            target = rng.choice(candidates)
            return {"numbers": nums, "target": target}


def main() -> None:
    rng = random.Random(42)
    rows = [gen_problem(rng) for _ in range(2000)]
    with open("countdown_train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} problems")


if __name__ == "__main__":
    main()
