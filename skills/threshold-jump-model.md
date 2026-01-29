# Threshold Jump Model Skill

## 理論的背景

### 閾値ジャンプ効果とは

消費者が特定の金額閾値（threshold）を超えることで得られるインセンティブ（送料無料等）のために、購入金額を意図的に引き上げる行動。

```
┌─────────────────────────────────────────────────────────────┐
│                    ジャンプ効果のメカニズム                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  現在のカート: 12,000円    閾値: 20,000円                    │
│                                                             │
│       「あと8,000円で送料無料」                               │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            心理的会計 (Mental Accounting)            │   │
│  │                                                     │   │
│  │  選択肢A: 送料1,500円を払う                          │   │
│  │    → 1,500円の「損失」として認識                     │   │
│  │                                                     │   │
│  │  選択肢B: 8,000円分追加購入して送料無料               │   │
│  │    → 8,000円の「商品を得る」として認識（利得）        │   │
│  │                                                     │   │
│  │  ⇒ 損失回避により、選択肢Bが選好される               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### エビデンスサマリー

| 出典 | 発見 | 信頼度 |
|------|------|--------|
| Shopify調査 (2026) | 58%が送料無料のため追加購入検討 | 高 |
| Shopify調査 (2026) | 80%が最低購入額を満たそうとする | 高 |
| 業界メタ分析 | 送料無料閾値でAOV 30%増 | 中 |
| Statista (2025) | 39%が送料でカート離脱 | 高 |
| RAND実験 (1982) | 閾値超過で行動30%変化 | 高 |

---

## シミュレーションモデル

### モデル構造

```python
def simulate_threshold_jump(
    n_customers: int,
    current_distribution: dict,  # 現在の価格帯分布
    threshold: float,            # 送料無料閾値
    params: dict,                # 行動パラメータ
    seed: int = 42
) -> SimulationResult:
    """
    閾値ジャンプシミュレーション
    
    Returns:
        SimulationResult: シミュレーション結果
            - new_distribution: 新しい価格帯分布
            - jump_rate: ジャンプ率
            - aov_change: AOV変化
            - revenue_impact: 売上影響
            - margin_impact: 粗利影響
    """
```

### ジャンプ確率の計算

```python
def calculate_jump_probability(
    current_amount: float,
    threshold: float,
    params: dict
) -> float:
    """
    個別顧客のジャンプ確率を計算
    
    P(jump) = base_propensity × distance_factor × segment_factor
    
    Parameters:
        current_amount: 現在のカート金額
        threshold: 送料無料閾値
        params: {
            'base_propensity': 0.58,  # 基本ジャンプ傾向
            'goal_gradient': 1.5,     # 目標勾配係数
            'segment_factor': 1.0     # セグメント係数
        }
    
    Returns:
        float: ジャンプ確率 (0-1)
    """
    gap = threshold - current_amount
    gap_ratio = gap / current_amount
    
    # 距離要因: 閾値に近いほど高い（Goal Gradient Effect）
    # gap_ratioが小さいほどジャンプしやすい
    if gap_ratio > 1.0:
        distance_factor = 0.3  # 2倍以上離れていると難しい
    elif gap_ratio > 0.5:
        distance_factor = 0.5  # 1.5倍以上離れている
    else:
        distance_factor = 1.0 - (gap_ratio * 0.5)  # 近いほど高い
    
    probability = (
        params['base_propensity'] 
        * distance_factor 
        * params['segment_factor']
    )
    
    return min(probability, 0.95)  # 上限95%
```

### ジャンプ後の金額決定

```python
def determine_jump_amount(
    current_amount: float,
    threshold: float,
    params: dict
) -> float:
    """
    ジャンプ後の注文金額を決定
    
    多くの顧客は閾値ちょうどではなく、やや超える金額で購入。
    
    Parameters:
        current_amount: 現在のカート金額
        threshold: 閾値
        params: {
            'overspend_mean': 0.10,   # 平均超過率 10%
            'overspend_std': 0.05     # 標準偏差 5%
        }
    
    Returns:
        float: ジャンプ後の注文金額
    """
    # 正規分布で超過率を決定
    overspend_rate = np.random.normal(
        params['overspend_mean'],
        params['overspend_std']
    )
    overspend_rate = max(0, overspend_rate)  # 負の値は0に
    
    return threshold * (1 + overspend_rate)
```

---

## パラメータ定義

### 行動パラメータ

| パラメータ | 記号 | デフォルト | 範囲 | 根拠 |
|-----------|------|-----------|------|------|
| 基本ジャンプ傾向 | base_propensity | 0.58 | 0.4-0.7 | Shopify調査 |
| 目標勾配係数 | goal_gradient | 1.5 | 1.0-2.0 | 行動経済学 |
| 超過率平均 | overspend_mean | 0.10 | 0.05-0.20 | 推定 |
| 超過率標準偏差 | overspend_std | 0.05 | 0.03-0.10 | 推定 |

### セグメント係数

| セグメント | 係数 | 根拠 |
|-----------|------|------|
| toC（個人） | 1.0 | 基準 |
| toB（法人） | 0.8 | 価格感応度低い |
| リピーター | 1.2 | ブランド信頼あり |
| 新規 | 0.9 | 様子見傾向 |

---

## 感度分析パラメータ

| パラメータ | 低位 | 中位 | 高位 | 影響度 |
|-----------|------|------|------|--------|
| base_propensity | 0.40 | 0.58 | 0.70 | **高** |
| overspend_mean | 0.05 | 0.10 | 0.20 | 中 |
| 粗利率 | 0.25 | 0.30 | 0.35 | **高** |
| 送料コスト | 1,000 | 1,350 | 1,500 | 中 |

---

## 出力仕様

### SimulationResult

```python
@dataclass
class SimulationResult:
    # 分布変化
    before_distribution: Dict[str, float]
    after_distribution: Dict[str, float]
    
    # 主要指標
    jump_rate: float              # ジャンプ率
    aov_before: float             # AOV（変更前）
    aov_after: float              # AOV（変更後）
    aov_change_rate: float        # AOV変化率
    
    # 収益影響
    revenue_before: float
    revenue_after: float
    revenue_change: float
    
    margin_before: float
    margin_after: float
    margin_change: float
    
    # 詳細データ
    customer_level_data: pd.DataFrame
```

---

*このスキルは skills/simulation-workflow.md と組み合わせて使用*
