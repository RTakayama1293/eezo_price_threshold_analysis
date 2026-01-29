# EEZO 価格帯ジャンプ効果シミュレーター

## 概要

EEZO ECサイトのShopifyリニューアルに向けた、価格帯ジャンプ効果のシミュレーターです。

**核心的な問い**:  
> 「送料無料閾値を20,000円に設定することで、10,000円帯の顧客を20,000円帯へジャンプさせられるか？」

## 背景

行動経済学の研究によると、EC サイトにおける送料無料閾値は消費者行動に大きな影響を与えます。

- 58% の消費者が送料無料のために追加購入を検討（Shopify調査）
- 送料無料閾値設定で AOV が 30% 増加（業界調査）
- 39% が送料を理由にカート離脱（Statista）

本シミュレーターは、これらのエビデンスに基づき、EEZO 固有の状況で効果を試算します。

## クイックスタート

### 1. Claude Code on the Web で開く

1. このリポジトリを GitHub にプッシュ
2. https://claude.ai/code でリポジトリを選択
3. 以下の指示を実行:

```
シミュレーションを実行して
```

### 2. ローカル実行（オプション）

```bash
pip install -r requirements.txt
python src/simulator.py
```

## ディレクトリ構成

```
snj-eezo-threshold-jump-simulator/
├── CLAUDE.md                    # プロジェクト指示書（Claude Code用）
├── README.md                    # 本ファイル
├── requirements.txt             # Python依存パッケージ
├── .gitignore
├── .claude/
│   └── settings.json            # 自動セットアップフック
├── rules/
│   └── simulation.md            # シミュレーションルール
├── skills/
│   ├── threshold-jump-model.md  # 閾値ジャンプモデル定義
│   ├── simulation-workflow.md   # シミュレーションワークフロー
│   └── domain-knowledge.md      # EEZOドメイン知識
├── data/
│   ├── raw/                     # パラメータ設定（編集禁止）
│   │   ├── behavior_params.json # 顧客行動パラメータ
│   │   └── product_tiers.json   # 商品価格帯設計
│   └── processed/               # 加工済みデータ
├── experiments/
│   └── exp001_baseline/         # 実験ディレクトリ
│       └── outputs/
├── src/
│   ├── simulator.py             # シミュレーター本体
│   ├── visualizer.py            # 可視化モジュール
│   └── utils.py                 # ユーティリティ
└── outputs/
    ├── reports/                 # レポート出力先
    └── figures/                 # 図表出力先
```

## 主要なシナリオ

| シナリオ | 説明 |
|---------|------|
| baseline | 現状維持（送料一律課金） |
| threshold_20k | 20,000円以上で送料無料 |
| tiered | 段階的閾値（10K:割引、20K:無料） |

## 出力例

シミュレーション結果サマリー:

| 指標 | Before | After | 変化 |
|------|--------|-------|------|
| 平均AOV | ¥12,500 | ¥16,250 | +30% |
| ジャンプ率 | - | 35% | - |
| 売上総額 | ¥1,000万 | ¥1,300万 | +30% |
| 粗利総額 | ¥300万 | ¥360万 | +20% |

## 関連ドキュメント

- [CLAUDE.md](./CLAUDE.md) - Claude Code 用プロジェクト指示書
- [skills/threshold-jump-model.md](./skills/threshold-jump-model.md) - 閾値ジャンプモデルの詳細
- [skills/simulation-workflow.md](./skills/simulation-workflow.md) - シミュレーションワークフロー

## ライセンス

新日本海商事 内部利用限定

---

*作成日: 2026-01-29*
