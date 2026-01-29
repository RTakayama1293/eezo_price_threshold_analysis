"""
EEZO 価格帯ジャンプ効果シミュレーター

10,000円帯 → 20,000円帯へのジャンプ効果をシミュレーション
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class SimulationResult:
    """シミュレーション結果を格納するデータクラス"""
    
    scenario_name: str
    seed: int
    n_customers: int
    threshold: Optional[float]
    
    # 分布変化
    before_distribution: Dict[str, float] = field(default_factory=dict)
    after_distribution: Dict[str, float] = field(default_factory=dict)
    
    # 主要指標
    jump_rate: float = 0.0
    aov_before: float = 0.0
    aov_after: float = 0.0
    aov_change_rate: float = 0.0
    
    # 収益影響
    revenue_before: float = 0.0
    revenue_after: float = 0.0
    revenue_change: float = 0.0
    revenue_change_rate: float = 0.0
    
    margin_before: float = 0.0
    margin_after: float = 0.0
    margin_change: float = 0.0
    margin_change_rate: float = 0.0
    
    # 詳細データ
    customer_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    def summary(self) -> str:
        """結果サマリーを文字列で返す"""
        return f"""
=== シミュレーション結果: {self.scenario_name} ===

【設定】
- 顧客数: {self.n_customers:,}人
- 閾値: {f'¥{self.threshold:,.0f}' if self.threshold else 'なし'}
- 乱数シード: {self.seed}

【主要指標】
- ジャンプ率: {self.jump_rate:.1%}
- AOV変化: ¥{self.aov_before:,.0f} → ¥{self.aov_after:,.0f} ({self.aov_change_rate:+.1%})

【収益影響】
- 売上: ¥{self.revenue_before:,.0f} → ¥{self.revenue_after:,.0f} ({self.revenue_change_rate:+.1%})
- 粗利: ¥{self.margin_before:,.0f} → ¥{self.margin_after:,.0f} ({self.margin_change_rate:+.1%})

【価格帯分布変化】
Before → After
"""
    
    def save(self, path: str):
        """結果をCSVに保存"""
        self.customer_data.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"結果を保存しました: {path}")


class ThresholdJumpSimulator:
    """閾値ジャンプ効果シミュレーター"""
    
    def __init__(
        self,
        params_path: str = 'data/raw/behavior_params.json',
        tiers_path: str = 'data/raw/product_tiers.json',
        seed: int = 42
    ):
        """
        シミュレーターを初期化
        
        Args:
            params_path: 行動パラメータファイルのパス
            tiers_path: 価格帯設定ファイルのパス
            seed: 乱数シード
        """
        self.seed = seed
        np.random.seed(seed)
        
        # パラメータ読み込み
        self.params = self._load_params(params_path)
        self.tiers = self._load_tiers(tiers_path)
        
    def _load_params(self, path: str) -> Dict:
        """行動パラメータを読み込み"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: {path} が見つかりません。デフォルト値を使用します。")
            return self._default_params()
    
    def _load_tiers(self, path: str) -> Dict:
        """価格帯設定を読み込み"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: {path} が見つかりません。デフォルト値を使用します。")
            return self._default_tiers()
    
    def _default_params(self) -> Dict:
        """デフォルトの行動パラメータ"""
        return {
            'base_propensity': {'value': 0.58},
            'jump_success_rate': {'value': 0.80},
            'overspend_mean': {'value': 0.10},
            'overspend_std': {'value': 0.05}
        }
    
    def _default_tiers(self) -> Dict:
        """デフォルトの価格帯設定"""
        return {
            'current_distribution': {
                'tiers': {
                    'tier_0_5000': {'share': 0.10, 'avg_amount': 3500, 'margin_rate': 0.25},
                    'tier_5000_10000': {'share': 0.35, 'avg_amount': 7500, 'margin_rate': 0.28},
                    'tier_10000_15000': {'share': 0.30, 'avg_amount': 12500, 'margin_rate': 0.30},
                    'tier_15000_20000': {'share': 0.15, 'avg_amount': 17500, 'margin_rate': 0.32},
                    'tier_20000_plus': {'share': 0.10, 'avg_amount': 28000, 'margin_rate': 0.35}
                }
            },
            'shipping_costs': {
                'average_shipping_cost': 1350,
                'current_shipping_fee': 1500
            }
        }
    
    def _generate_customers(self, n_customers: int) -> pd.DataFrame:
        """顧客データを生成"""
        tiers_config = self.tiers['current_distribution']['tiers']
        
        customers = []
        for tier_name, tier_info in tiers_config.items():
            n_tier = int(n_customers * tier_info['share'])
            
            # 価格帯内で正規分布に基づく金額を生成
            avg = tier_info['avg_amount']
            std = avg * 0.15  # 標準偏差は平均の15%
            
            amounts = np.random.normal(avg, std, n_tier)
            amounts = np.clip(amounts, avg * 0.7, avg * 1.3)  # 範囲制限
            
            for amount in amounts:
                customers.append({
                    'original_tier': tier_name,
                    'original_amount': amount,
                    'margin_rate': tier_info['margin_rate']
                })
        
        return pd.DataFrame(customers)
    
    def _calculate_jump_probability(
        self,
        current_amount: float,
        threshold: float
    ) -> float:
        """ジャンプ確率を計算"""
        if current_amount >= threshold:
            return 0.0  # 既に閾値以上
        
        gap = threshold - current_amount
        gap_ratio = gap / current_amount
        
        # 距離要因
        if gap_ratio > 1.0:
            distance_factor = 0.3
        elif gap_ratio > 0.5:
            distance_factor = 0.5
        else:
            distance_factor = 1.0 - (gap_ratio * 0.5)
        
        base_propensity = self.params['base_propensity']['value']
        probability = base_propensity * distance_factor
        
        return min(probability, 0.95)
    
    def _determine_jump_amount(self, threshold: float) -> float:
        """ジャンプ後の金額を決定"""
        overspend_mean = self.params['overspend_mean']['value']
        overspend_std = self.params['overspend_std']['value']
        
        overspend_rate = np.random.normal(overspend_mean, overspend_std)
        overspend_rate = max(0, overspend_rate)
        
        return threshold * (1 + overspend_rate)
    
    def run(
        self,
        scenario_name: str,
        threshold: Optional[float],
        shipping_fee_above: float = 0,
        shipping_fee_below: float = 1500,
        n_customers: int = 1000
    ) -> SimulationResult:
        """
        シミュレーションを実行
        
        Args:
            scenario_name: シナリオ名
            threshold: 送料無料閾値（Noneの場合は閾値なし）
            shipping_fee_above: 閾値以上の送料
            shipping_fee_below: 閾値未満の送料
            n_customers: シミュレーション顧客数
        
        Returns:
            SimulationResult: シミュレーション結果
        """
        np.random.seed(self.seed)
        
        # 顧客データ生成
        customers = self._generate_customers(n_customers)
        
        # ベースライン計算
        customers['before_amount'] = customers['original_amount']
        customers['before_shipping'] = shipping_fee_below
        customers['before_total'] = customers['before_amount'] + customers['before_shipping']
        customers['before_margin'] = customers['before_amount'] * customers['margin_rate']
        
        # 閾値シナリオの場合、ジャンプを計算
        if threshold:
            customers['jump_probability'] = customers['original_amount'].apply(
                lambda x: self._calculate_jump_probability(x, threshold)
            )
            customers['jump_decision'] = np.random.random(len(customers)) < customers['jump_probability']
            
            def calc_after_amount(row):
                if row['original_amount'] >= threshold:
                    return row['original_amount']
                elif row['jump_decision']:
                    return self._determine_jump_amount(threshold)
                else:
                    return row['original_amount']
            
            customers['after_amount'] = customers.apply(calc_after_amount, axis=1)
            customers['after_shipping'] = customers['after_amount'].apply(
                lambda x: shipping_fee_above if x >= threshold else shipping_fee_below
            )
        else:
            customers['jump_probability'] = 0.0
            customers['jump_decision'] = False
            customers['after_amount'] = customers['original_amount']
            customers['after_shipping'] = shipping_fee_below
        
        customers['after_total'] = customers['after_amount'] + customers['after_shipping']
        customers['after_margin'] = customers['after_amount'] * customers['margin_rate']
        
        # 価格帯を再分類
        def classify_tier(amount):
            if amount < 5000:
                return 'tier_0_5000'
            elif amount < 10000:
                return 'tier_5000_10000'
            elif amount < 15000:
                return 'tier_10000_15000'
            elif amount < 20000:
                return 'tier_15000_20000'
            else:
                return 'tier_20000_plus'
        
        customers['after_tier'] = customers['after_amount'].apply(classify_tier)
        
        # 結果集計
        result = SimulationResult(
            scenario_name=scenario_name,
            seed=self.seed,
            n_customers=n_customers,
            threshold=threshold
        )
        
        # 分布変化
        result.before_distribution = customers['original_tier'].value_counts(normalize=True).to_dict()
        result.after_distribution = customers['after_tier'].value_counts(normalize=True).to_dict()
        
        # ジャンプ対象（10,000円～20,000円帯）のジャンプ率
        jump_candidates = customers[
            (customers['original_amount'] >= 10000) & 
            (customers['original_amount'] < 20000)
        ]
        if len(jump_candidates) > 0:
            result.jump_rate = jump_candidates['jump_decision'].mean()
        
        # AOV
        result.aov_before = customers['before_amount'].mean()
        result.aov_after = customers['after_amount'].mean()
        result.aov_change_rate = (result.aov_after - result.aov_before) / result.aov_before
        
        # 収益
        result.revenue_before = customers['before_amount'].sum()
        result.revenue_after = customers['after_amount'].sum()
        result.revenue_change = result.revenue_after - result.revenue_before
        result.revenue_change_rate = result.revenue_change / result.revenue_before
        
        # 粗利（送料粗利も考慮）
        shipping_cost = self.tiers['shipping_costs']['average_shipping_cost']
        
        before_shipping_margin = (customers['before_shipping'] - shipping_cost).sum()
        after_shipping_margin = (customers['after_shipping'] - shipping_cost).sum()
        
        result.margin_before = customers['before_margin'].sum() + before_shipping_margin
        result.margin_after = customers['after_margin'].sum() + after_shipping_margin
        result.margin_change = result.margin_after - result.margin_before
        result.margin_change_rate = result.margin_change / result.margin_before
        
        # 顧客データを保存
        result.customer_data = customers
        
        return result


class SensitivityAnalyzer:
    """感度分析を実行するクラス"""
    
    def __init__(self, simulator: ThresholdJumpSimulator):
        self.simulator = simulator
    
    def run(
        self,
        parameter: str,
        values: List[float],
        base_scenario: str = 'threshold_20k',
        threshold: float = 20000,
        n_customers: int = 1000
    ) -> pd.DataFrame:
        """
        感度分析を実行
        
        Args:
            parameter: 分析対象パラメータ名
            values: パラメータ値のリスト
            base_scenario: ベースシナリオ名
            threshold: 閾値
            n_customers: 顧客数
        
        Returns:
            pd.DataFrame: 感度分析結果
        """
        results = []
        original_value = self.simulator.params[parameter]['value']
        
        for value in values:
            # パラメータを一時的に変更
            self.simulator.params[parameter]['value'] = value
            
            # シミュレーション実行
            result = self.simulator.run(
                scenario_name=f'{base_scenario}_{parameter}_{value}',
                threshold=threshold,
                n_customers=n_customers
            )
            
            results.append({
                'parameter': parameter,
                'value': value,
                'jump_rate': result.jump_rate,
                'aov_change_rate': result.aov_change_rate,
                'revenue_change_rate': result.revenue_change_rate,
                'margin_change_rate': result.margin_change_rate
            })
        
        # 元の値に戻す
        self.simulator.params[parameter]['value'] = original_value
        
        return pd.DataFrame(results)


def main():
    """メイン実行"""
    print("=== EEZO 価格帯ジャンプ効果シミュレーター ===\n")
    
    # シミュレーター初期化
    sim = ThresholdJumpSimulator(seed=42)
    
    # ベースライン実行
    print("【ベースラインシナリオ】")
    baseline = sim.run(
        scenario_name='baseline',
        threshold=None,
        n_customers=1000
    )
    print(baseline.summary())
    
    # 20,000円閾値シナリオ
    print("\n【20,000円閾値シナリオ】")
    threshold_20k = sim.run(
        scenario_name='threshold_20k',
        threshold=20000,
        n_customers=1000
    )
    print(threshold_20k.summary())
    
    # 比較
    print("\n【シナリオ比較】")
    print(f"AOV変化: {baseline.aov_before:,.0f} → {threshold_20k.aov_after:,.0f} "
          f"({threshold_20k.aov_change_rate:+.1%})")
    print(f"ジャンプ率: {threshold_20k.jump_rate:.1%}")
    print(f"粗利変化: {threshold_20k.margin_change_rate:+.1%}")


if __name__ == '__main__':
    main()
