# Open-R1で学ぶLLMアーキテクチャ＆学習 チュートリアル

DeepSeek-R1 を完全オープンに再現するHugging Faceの **Open-R1** プロジェクトを題材に、
大規模言語モデル（LLM）の **アーキテクチャ** と **学習手法（SFT / RL / GRPO / 蒸留）** を
手を動かしながら学ぶ日本語教科書です。

> 参考にした元記事: [Open-R1: a fully open reproduction of DeepSeek-R1 (Hugging Face Blog)](https://huggingface.co/blog/open-r1)
> リポジトリ: [huggingface/open-r1](https://github.com/huggingface/open-r1)
> 本教科書の構成は [DrRacket-Japanese-Tutorial](https://github.com/hama-jp/DrRacket-Japanese-Tutorial) の形式を踏襲しています。

---

## 🎯 この教科書の狙い

- **DeepSeek-R1 / Open-R1** の技術レポートを読み解くための前提知識を体系化する
- Transformer・MoE・RoPE など **最新LLMの構成要素** をひとつずつ理解する
- SFT → RLHF → GRPO → 蒸留 という **推論モデルの学習パイプライン** を追体験する
- 最後に実際に `open-r1` リポジトリを動かし、**自分で小さな推論モデルを育ててみる**

Python・機械学習の基礎（`torch` を少し触ったことがある）を前提にしますが、
強化学習や推論モデルに踏み込むのは初めて、という読者を主なターゲットにしています。

## 📚 目次

### Part 0 — はじめに

| 章 | タイトル | 概要 |
|---|---|---|
| [0](book/ch00.md) | まえがき・本書の読み方 | 対象読者・記法・推奨学習順 |
| [1](book/ch01.md) | LLM と Open-R1 の全体像 | R1/R1-Zero・3ステップ再現計画を俯瞰 |

### Part 1 — LLMアーキテクチャの基礎

| 章 | タイトル | 概要 |
|---|---|---|
| [2](book/ch02.md) | Transformer の骨格 | デコーダブロック・Attention・FFN |
| [3](book/ch03.md) | Mixture of Experts (MoE) | DeepSeek-V3 の671B/37Bを支える仕組み |
| [4](book/ch04.md) | 位置表現と RoPE | 位置エンコーディングと長文対応 |

### Part 2 — LLMの学習プロセス

| 章 | タイトル | 概要 |
|---|---|---|
| [5](book/ch05.md) | 事前学習とSFT | 次トークン予測と教師あり微調整 |
| [6](book/ch06.md) | 強化学習入門（RLHF/PPO） | 報酬モデル・方策勾配の基礎 |

### Part 3 — Open-R1 のアプローチ

| 章 | タイトル | 概要 |
|---|---|---|
| [7](book/ch07.md) | GRPO の数理 | 価値関数不要のグループベースRL |
| [8](book/ch08.md) | ルールベース報酬と "あはもーめんと" | 書式報酬・正解報酬・推論の創発 |
| [9](book/ch09.md) | 蒸留で小さな推論モデルを作る | R1-Distill とは何か |
| [10](book/ch10.md) | データセットと評価 | OpenR1-Math / CodeForces-CoTs / AIME |

### Part 4 — 手を動かして学ぶ

| 章 | タイトル | 概要 |
|---|---|---|
| [11](book/ch11.md) | 環境構築とミニチュア実験 | `uv` と `trl`・小さなモデルで試す |
| [12](book/ch12.md) | 自分でGRPOを回してみる | Countdown ゲームで "あはもーめんと"を観測 |

### 付録

- [付録A: 参考文献とさらなる学習資源](book/appendix.md)
- [演習解答集](solutions/README.md)

## 🗂️ ディレクトリ構成

```
Open-R1_Tutorial/
├── README.md           # この文書
├── book/               # 本文（章ごとにMarkdown）
│   ├── ch00.md ～ ch12.md
│   └── appendix.md
├── examples/           # 動かして試すコード
│   └── chNN/           # 章番号ごと
├── solutions/          # 演習解答
│   └── chNN.md
└── images/             # 図版
```

## 💻 動作環境

本書のコード例は以下で動作確認しています。

| ソフトウェア | バージョン |
|---|---|
| Python | 3.11 以上 |
| CUDA | 12.1 以上 |
| PyTorch | 2.3 以上 |
| transformers | 4.45 以上 |
| trl | 0.14 以上 |
| accelerate | 1.0 以上 |
| vllm | 0.6 以上（任意） |

GPUを持たない環境でも、**11章** までは CPU あるいは Google Colab の T4 GPU で概ね読み進められます。
12章以降の GRPO 実験は、少なくとも 1 枚の 24GB クラスGPU（RTX 3090 / 4090 など）を推奨します。

## 📝 記法

本書では次の記法を用います。

- コードブロック先頭の `$` はシェルコマンド、`>>>` は Python REPL を表します
- **太字** は用語の初出、`等幅` はコード・ファイル名・ハイパーパラメータ名
- 💡 は補足、⚠️ は注意、🧪 は「手を動かしてみよう」の演習マーカーです

```bash
$ echo "シェルコマンドの例"
```

```python
>>> import torch
>>> torch.cuda.is_available()
True
```

## 🤝 貢献・フィードバック

間違いや「ここがわかりにくい」という声は Issue / Pull Request で歓迎します。
誤字脱字レベルの修正も、推論モデル理解の裾野を広げる大事な貢献です。

## 📄 ライセンス

本文・コード共に **MIT License** で公開します。商用利用・改変・再配布も自由です。
元ネタである [huggingface/open-r1](https://github.com/huggingface/open-r1)（Apache-2.0）
のライセンスにも合わせて従ってください。
