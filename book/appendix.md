# 付録A 参考文献とさらなる学習資源

本書で扱いきれなかった話題や、より深く知りたい読者のための入り口をまとめます。

## A.1 元論文・1次資料

### DeepSeek

- **DeepSeek-R1 technical report** — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
  本書第1章・7〜9章の元ネタ。
- **DeepSeek-V3 technical report** — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
  V3 の MoE 構造・MLA・学習レシピ。第2〜3章の補強に。
- **DeepSeekMath (GRPO 原論文)** — [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
  GRPO がもともと提案されたのは数学推論論文。

### Open-R1

- **Blog: Open-R1** — https://huggingface.co/blog/open-r1
- **Blog: Update #1** — https://huggingface.co/blog/open-r1/update-1
- **Blog: Update #2** — https://huggingface.co/blog/open-r1/update-2
- **Blog: Mini-R1 (Countdown)** — https://huggingface.co/blog/open-r1/mini-r1-contdown-game
- **GitHub: huggingface/open-r1** — https://github.com/huggingface/open-r1

### Transformer / LLM 基礎

- **Attention Is All You Need** — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **LLaMA** — [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)（RMSNorm, SwiGLU, RoPE の採用）
- **RoFormer (RoPE)** — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- **YaRN** — [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
- **Switch Transformer / GLaM (MoE)** — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)

### 強化学習 / RLHF

- **PPO 原論文** — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- **InstructGPT** — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) (RLHF 3 ステップの元ネタ)
- **Constitutional AI** — [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- **DPO (PPO を外す代替)** — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

## A.2 実装リポジトリ

| 目的 | リポジトリ |
|---|---|
| RL・SFT の統合ライブラリ | [huggingface/trl](https://github.com/huggingface/trl) |
| 推論最適化 | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| 評価 | [huggingface/lighteval](https://github.com/huggingface/lighteval) |
| 推論モデル学習（別実装） | [volcengine/verl](https://github.com/volcengine/verl) |
| 軽量GRPO実装 | [dvruette/grpo-quickstart](https://github.com/huggingface/trl) の `examples/` |
| OpenRLHF | [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) |

## A.3 データセット

- [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) — 数学推論の王道
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) — 算数文章題の古典
- [MATH](https://huggingface.co/datasets/lighteval/MATH) — 競技数学
- [CodeContests](https://huggingface.co/datasets/deepmind/code_contests) — 競プロ
- [AIME 2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) — 米国数学オリンピック予選
- [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) — 博士号レベル理系 QA
- [LiveCodeBench](https://huggingface.co/datasets/livecodebench/code_generation_lite) — 継続更新される competitive coding

## A.4 書籍・講義

- *Deep Learning* (Goodfellow, Bengio, Courville) — DL の基礎
- *Reinforcement Learning: An Introduction* (Sutton, Barto, 2nd ed.) — RL の古典
- Stanford CS224N / CS336 — NLP / 基盤モデルの講義録
- Chip Huyen ブログ *"RLHF and its alternatives"* — 概観に最適

## A.5 本書で扱えなかったトピック

- **DPO / KTO** — 報酬モデル不要の新しい Preference 最適化
- **Test-time scaling** — 推論時にサンプル数で性能を伸ばすテクニック
- **Self-consistency / Best-of-N / Tree-of-Thoughts** — CoT の探索拡張
- **Process Reward Model (PRM)** — トークン・ステップ単位の報酬
- **LoRA / QLoRA** — 計算資源を節約する微調整
- **マルチモーダル推論モデル** — Vision + 推論
- **長文対応推論（128K, 1M）** — RingAttention, RoPE scaling

興味があるテーマから派生して掘り下げてください。
そして、ぜひあなたの手で、次の推論モデルを作ってみましょう。

---

[← 第12章 実践GRPO](ch12.md) ｜ [トップに戻る](../README.md)
