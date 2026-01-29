# Simulation Workflow Skill

## 標準シミュレーションフロー

### Step 1: 仮説の明文化

実験ディレクトリに `hypothesis.md` を作成:

```markdown
## 仮説

### メイン仮説
送料無料閾値を20,000円に設定することで、
10,000円～15,000円帯の顧客の30%以上が20,000円帯へジャンプする。

### サブ仮説
1. ジャンプ顧客のAOVは閾値を10%程度超える（22,000円前後）
2. 送料粗利の減少分は、AOV増による粗利増で相殺可能
3. 「あと○○円で送料無料」表示が効果を最大化する

### 検証方法
- モンテカルロシミュレーション（N=1,000顧客 × 100回）
- 感度分析（ジャンプ率パラメータ）
```

### Step 2: パラメータ設定

```python
import json

# パラメータ読み込み
with open('data/raw/behavior_params.json') as f:
    params = json.load(f)

# 根拠確認
print("=== パラメータと根拠 ===")
for key, value in params.items():
    print(f"{key}: {value['value']}")
    print(f"  根拠: {value['source']}")
    print(f"  範囲: {value['range']}")
    print()
```

### Step 3: ベースラインシナリオ実行

```python
from src.simulator import ThresholdJumpSimulator

# シミュレーター初期化
sim = ThresholdJumpSimulator(seed=42)

# ベースライン（閾値なし）
baseline = sim.run(
    scenario_name='baseline',
    threshold=None,
    shipping_fee=1500,
    n_customers=1000
)

# 結果保存
baseline.save('experiments/exp001_baseline/outputs/baseline_result.csv')
print(baseline.summary())
```

### Step 4: 閾値シナリオ実行

```python
# 20,000円閾値シナリオ
threshold_20k = sim.run(
    scenario_name='threshold_20k',
    threshold=20000,
    shipping_fee_above=0,
    shipping_fee_below=1500,
    n_customers=1000
)

# 結果保存
threshold_20k.save('experiments/exp001_baseline/outputs/threshold_20k_result.csv')
print(threshold_20k.summary())
```

### Step 5: 感度分析

```python
from src.simulator import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(sim)

# ジャンプ率パラメータの感度分析
sensitivity = analyzer.run(
    parameter='base_propensity',
    values=[0.40, 0.45, 0.50, 0.55, 0.58, 0.65, 0.70],
    base_scenario='threshold_20k'
)

sensitivity.save('experiments/exp001_baseline/outputs/sensitivity_propensity.csv')
sensitivity.plot('outputs/figures/sensitivity_propensity.png')
```

### Step 6: 結果の可視化

```python
from src.visualizer import Visualizer

viz = Visualizer()

# 1. 価格帯分布の変化（Before/After）
viz.plot_distribution_shift(
    baseline=baseline,
    treatment=threshold_20k,
    title='価格帯分布の変化（20K閾値導入）',
    save_path='outputs/figures/distribution_shift.png'
)

# 2. ジャンプフロー（サンキー図）
viz.plot_jump_flow(
    result=threshold_20k,
    save_path='outputs/figures/jump_flow.png'
)

# 3. 収益インパクト
viz.plot_revenue_impact(
    baseline=baseline,
    treatment=threshold_20k,
    save_path='outputs/figures/revenue_impact.png'
)

# 4. 感度分析結果
viz.plot_sensitivity(
    sensitivity=sensitivity,
    save_path='outputs/figures/sensitivity_chart.png'
)
```

### Step 7: 示唆の抽出

```python
from src.analyzer import InsightExtractor

extractor = InsightExtractor()

insights = extractor.analyze(
    baseline=baseline,
    treatment=threshold_20k,
    sensitivity=sensitivity
)

print(insights.summary())
```

期待される出力:
```
=== 主要インサイト ===

1. ジャンプ効果
   - 10,000円～15,000円帯の35%が20,000円帯へジャンプ
   - 平均ジャンプ金額: +8,500円

2. AOVへの影響
   - AOV: ¥12,500 → ¥16,250 (+30%)
   - 業界ベンチマーク（30%増）と整合

3. 収益インパクト
   - 売上: +30%
   - 粗利: +20%（送料粗利減を考慮）
   - 損益分岐ジャンプ率: 25%（現在35%で十分）

4. リスク評価
   - ジャンプ率が40%→55%の場合も利益増
   - 最悪ケース（25%）でも損失なし

5. 商品構成への示唆
   - 8,000円～10,000円の「追加購入しやすい商品」が重要
   - セット商品で20,000円ちょうどを狙う設計が有効
```

### Step 8: レポート作成

```python
from src.reporter import MarkdownReporter

reporter = MarkdownReporter()

report = reporter.generate(
    experiment_id='exp001',
    hypothesis_path='experiments/exp001_baseline/hypothesis.md',
    results={
        'baseline': baseline,
        'threshold_20k': threshold_20k
    },
    sensitivity=sensitivity,
    insights=insights,
    figures=[
        'distribution_shift.png',
        'jump_flow.png',
        'revenue_impact.png',
        'sensitivity_chart.png'
    ]
)

report.save('outputs/reports/exp001_report.md')
```

---

## 出力ファイル一覧

| ファイル | 内容 | 形式 |
|---------|------|------|
| hypothesis.md | 仮説の明文化 | Markdown |
| baseline_result.csv | ベースライン結果 | CSV |
| threshold_20k_result.csv | 閾値シナリオ結果 | CSV |
| sensitivity_propensity.csv | 感度分析結果 | CSV |
| distribution_shift.png | 分布変化図 | PNG |
| jump_flow.png | ジャンプフロー図 | PNG |
| revenue_impact.png | 収益インパクト図 | PNG |
| sensitivity_chart.png | 感度分析図 | PNG |
| exp001_report.md | 最終レポート | Markdown |

---

## クイック実行

Claude Code on the Web での実行例:

```
「ベースラインシミュレーションを実行して」
```

```
「20,000円閾値シナリオと比較して」
```

```
「ジャンプ率の感度分析をして」
```

```
「結果をレポートにまとめて」
```

---

*このスキルは skills/threshold-jump-model.md と組み合わせて使用*
