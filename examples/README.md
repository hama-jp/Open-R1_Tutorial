# examples/

各章で紹介したコード例・実験スクリプトを章番号ごとに置いています。
本文の 🧪 「手を動かしてみよう」 の参考実装としてご利用ください。

## 使い方

```bash
$ uv venv .venv --python 3.11
$ source .venv/bin/activate
$ uv pip install torch transformers trl datasets accelerate
$ python examples/ch02/tiny_transformer.py
```

章ごとに必要な依存が異なります。各ファイルの冒頭コメントで追加依存を指示しています。

## 章別一覧

| 章 | ファイル | 内容 |
|---|---|---|
| 2 | `ch02/tiny_transformer.py` | Single-block Transformer の最小実装 |
| 3 | `ch03/moe_toy.py` | Toy MoE とロードバランス観察 |
| 4 | `ch04/rope_demo.py` | RoPE のスコア行列可視化 |
| 5 | `ch05/minisft.py` | 10 件データでのミニ SFT |
| 6 | `ch06/cartpole_ppo.py` | CartPole で REINFORCE vs PPO |
| 7 | `ch07/grpo_length.py` | 応答長を 100 字に揃える GRPO |
| 8 | `ch08/multiply_grpo.py` | 2桁×2桁の掛け算 GRPO |
| 9 | `ch09/reject_sample.py` | 正解CoTだけを抽出する Rejection Sampling |
| 10 | `ch10/eval_aime1.py` | AIME 1 問を Distill で解く |
| 11 | `ch11/quick_infer.py` | Distill-1.5B で素早く推論 |
| 11 | `ch11/make_tiny_ds.py` | 足し算データ生成 |
| 11 | `ch11/tiny_sft.py` | Qwen2.5-0.5B の SFT |
| 11 | `ch11/compare.py` | 学習前後の比較 |
| 12 | `ch12/make_countdown.py` | Countdown 問題生成 |
| 12 | `ch12/grpo_countdown.py` | Countdown での GRPO |
| 12 | `ch12/soft_reward.py` | ソフト報酬バリアント |

各スクリプトは **動くことを優先した最小実装** です。
本番用途に使う際は、エラーハンドリングやログ記録を追加してください。
