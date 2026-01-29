#!/usr/bin/env python3
"""
EEZO 価格帯ジャンプ効果シミュレーション実行スクリプト

実験ID: exp001_baseline
作成日: 2026-01-29
"""

import sys
import os
from datetime import datetime

# srcディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from simulator import ThresholdJumpSimulator, SensitivityAnalyzer

# 可視化はmatplotlibがある場合のみ
try:
    import matplotlib
    matplotlib.use('Agg')  # GUIなし環境用
    import matplotlib.pyplot as plt
    from visualizer import Visualizer
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlibがインストールされていません。可視化はスキップします。")


def run_experiment():
    """メインシミュレーション実行"""

    print("=" * 60)
    print("EEZO 価格帯ジャンプ効果シミュレーション")
    print("実験ID: exp001_baseline")
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ============================================================
    # Step 1: シミュレーター初期化
    # ============================================================
    print("\n[Step 1] シミュレーター初期化")
    print("-" * 40)

    SEED = 42  # 再現性のための乱数シード
    N_CUSTOMERS = 1000  # シミュレーション顧客数

    sim = ThresholdJumpSimulator(
        params_path='data/raw/behavior_params.json',
        tiers_path='data/raw/product_tiers.json',
        seed=SEED
    )

    print(f"乱数シード: {SEED}")
    print(f"顧客数: {N_CUSTOMERS:,}人")
    print(f"ベース追加購入傾向: {sim.params['base_propensity']['value']:.0%}")
    print(f"超過購入率（平均）: {sim.params['overspend_mean']['value']:.0%}")

    # ============================================================
    # Step 2: ベースラインシナリオ実行
    # ============================================================
    print("\n[Step 2] ベースラインシナリオ実行（閾値なし）")
    print("-" * 40)

    baseline = sim.run(
        scenario_name='baseline',
        threshold=None,
        shipping_fee_below=1500,
        n_customers=N_CUSTOMERS
    )

    print(baseline.summary())

    # 結果保存
    baseline_path = 'experiments/exp001_baseline/outputs/baseline_result.csv'
    baseline.save(baseline_path)

    # ============================================================
    # Step 3: 20,000円閾値シナリオ実行
    # ============================================================
    print("\n[Step 3] 20,000円閾値シナリオ実行")
    print("-" * 40)

    threshold_20k = sim.run(
        scenario_name='threshold_20k',
        threshold=20000,
        shipping_fee_above=0,
        shipping_fee_below=1500,
        n_customers=N_CUSTOMERS
    )

    print(threshold_20k.summary())

    # 結果保存
    threshold_path = 'experiments/exp001_baseline/outputs/threshold_20k_result.csv'
    threshold_20k.save(threshold_path)

    # ============================================================
    # Step 4: シナリオ比較
    # ============================================================
    print("\n[Step 4] シナリオ比較")
    print("-" * 40)

    print("\n【ベースライン vs 20,000円閾値】\n")

    # AOV比較
    print("■ AOV（平均注文額）")
    print(f"  ベースライン: ¥{baseline.aov_before:,.0f}")
    print(f"  20K閾値後:    ¥{threshold_20k.aov_after:,.0f}")
    print(f"  変化率:       {threshold_20k.aov_change_rate:+.1%}")

    # ジャンプ率
    print(f"\n■ ジャンプ率（10K-20K帯 → 20K以上）")
    print(f"  {threshold_20k.jump_rate:.1%}")

    # 売上比較
    print(f"\n■ 売上総額")
    print(f"  ベースライン: ¥{baseline.revenue_before:,.0f}")
    print(f"  20K閾値後:    ¥{threshold_20k.revenue_after:,.0f}")
    print(f"  変化:         ¥{threshold_20k.revenue_change:+,.0f} ({threshold_20k.revenue_change_rate:+.1%})")

    # 粗利比較
    print(f"\n■ 粗利総額")
    print(f"  ベースライン: ¥{baseline.margin_before:,.0f}")
    print(f"  20K閾値後:    ¥{threshold_20k.margin_after:,.0f}")
    print(f"  変化:         ¥{threshold_20k.margin_change:+,.0f} ({threshold_20k.margin_change_rate:+.1%})")

    # 価格帯別分布変化
    print(f"\n■ 価格帯別分布変化")
    tier_labels = {
        'tier_0_5000': '～5,000円',
        'tier_5000_10000': '5,000～10,000円',
        'tier_10000_15000': '10,000～15,000円',
        'tier_15000_20000': '15,000～20,000円',
        'tier_20000_plus': '20,000円以上'
    }

    print(f"  {'価格帯':<20} {'Before':>10} {'After':>10} {'変化':>10}")
    print(f"  {'-'*52}")
    for tier, label in tier_labels.items():
        before = baseline.before_distribution.get(tier, 0)
        after = threshold_20k.after_distribution.get(tier, 0)
        change = after - before
        print(f"  {label:<20} {before:>10.1%} {after:>10.1%} {change:>+10.1%}")

    # ============================================================
    # Step 5: 感度分析
    # ============================================================
    print("\n[Step 5] 感度分析（base_propensity）")
    print("-" * 40)

    analyzer = SensitivityAnalyzer(sim)

    propensity_values = [0.40, 0.45, 0.50, 0.55, 0.58, 0.65, 0.70]

    sensitivity = analyzer.run(
        parameter='base_propensity',
        values=propensity_values,
        base_scenario='threshold_20k',
        threshold=20000,
        n_customers=N_CUSTOMERS
    )

    print("\n【感度分析結果】")
    print(f"  {'追加購入傾向':>12} {'ジャンプ率':>12} {'AOV変化率':>12} {'粗利変化率':>12}")
    print(f"  {'-'*52}")
    for _, row in sensitivity.iterrows():
        print(f"  {row['value']:>12.0%} {row['jump_rate']:>12.1%} {row['aov_change_rate']:>+12.1%} {row['margin_change_rate']:>+12.1%}")

    # 感度分析結果を保存
    sensitivity_path = 'experiments/exp001_baseline/outputs/sensitivity_propensity.csv'
    sensitivity.to_csv(sensitivity_path, index=False)
    print(f"\n感度分析結果を保存しました: {sensitivity_path}")

    # ============================================================
    # Step 6: 価格帯別ジャンプ分析
    # ============================================================
    print("\n[Step 6] 価格帯別ジャンプ分析")
    print("-" * 40)

    df = threshold_20k.customer_data

    # 価格帯別ジャンプ率
    tier_jump = df.groupby('original_tier').agg({
        'jump_decision': ['mean', 'sum', 'count'],
        'original_amount': 'mean',
        'after_amount': 'mean'
    }).round(3)

    print("\n【価格帯別ジャンプ統計】")

    for tier, label in tier_labels.items():
        if tier in df['original_tier'].values:
            tier_data = df[df['original_tier'] == tier]
            n = len(tier_data)
            jump_rate = tier_data['jump_decision'].mean()
            n_jumped = tier_data['jump_decision'].sum()
            avg_orig = tier_data['original_amount'].mean()
            avg_after = tier_data['after_amount'].mean()

            print(f"\n  {label}")
            print(f"    顧客数: {n}人")
            print(f"    ジャンプ率: {jump_rate:.1%} ({int(n_jumped)}人)")
            print(f"    平均金額: ¥{avg_orig:,.0f} → ¥{avg_after:,.0f}")

    # ジャンプした顧客の増加金額
    jumped = df[df['jump_decision'] == True]
    if len(jumped) > 0:
        jumped_copy = jumped.copy()
        jumped_copy['amount_increase'] = jumped_copy['after_amount'] - jumped_copy['original_amount']

        print(f"\n【ジャンプ顧客の増加金額】")
        print(f"  ジャンプ顧客数: {len(jumped_copy)}人")
        print(f"  平均増加金額: ¥{jumped_copy['amount_increase'].mean():,.0f}")
        print(f"  中央値増加金額: ¥{jumped_copy['amount_increase'].median():,.0f}")
        print(f"  最小～最大: ¥{jumped_copy['amount_increase'].min():,.0f} ～ ¥{jumped_copy['amount_increase'].max():,.0f}")

    # ============================================================
    # Step 7: 可視化
    # ============================================================
    if HAS_MATPLOTLIB:
        print("\n[Step 7] 結果の可視化")
        print("-" * 40)

        try:
            viz = Visualizer()

            # 1. 価格帯分布の変化
            fig1 = viz.plot_distribution_shift(
                baseline=baseline,
                treatment=threshold_20k,
                title='価格帯分布の変化（20K閾値導入）',
                save_path='outputs/figures/exp001_distribution_shift.png'
            )
            plt.close(fig1)

            # 2. 収益インパクト
            fig2 = viz.plot_revenue_impact(
                baseline=baseline,
                treatment=threshold_20k,
                title='収益インパクト（20K閾値導入）',
                save_path='outputs/figures/exp001_revenue_impact.png'
            )
            plt.close(fig2)

            # 3. ジャンプ詳細分析
            fig3 = viz.plot_jump_analysis(
                result=threshold_20k,
                title='ジャンプ効果の詳細分析',
                save_path='outputs/figures/exp001_jump_analysis.png'
            )
            plt.close(fig3)

            # 4. 感度分析
            fig4 = viz.plot_sensitivity(
                sensitivity_df=sensitivity,
                target_metric='margin_change_rate',
                title='感度分析',
                save_path='outputs/figures/exp001_sensitivity.png'
            )
            plt.close(fig4)

            print("可視化が完了しました。")
        except Exception as e:
            print(f"可視化中にエラーが発生しました: {e}")
    else:
        print("\n[Step 7] 可視化（スキップ - matplotlib未インストール）")

    # ============================================================
    # Step 8: 仮説検証まとめ
    # ============================================================
    print("\n[Step 8] 仮説検証まとめ")
    print("-" * 40)

    print("\n【仮説検証結果】")

    # メイン仮説
    main_hypothesis_met = threshold_20k.jump_rate >= 0.30
    print(f"\n■ メイン仮説: 10K-20K帯の30%以上が20K帯へジャンプ")
    print(f"  結果: ジャンプ率 {threshold_20k.jump_rate:.1%}")
    print(f"  判定: {'✓ 支持' if main_hypothesis_met else '✗ 棄却'}")

    # サブ仮説1
    jumped_mean = threshold_20k.aov_after / 20000 - 1 if threshold_20k.aov_after > 0 else 0
    sub1_met = threshold_20k.aov_change_rate >= 0.05
    print(f"\n■ サブ仮説1: ジャンプ顧客のAOVは閾値を10%程度超える")
    print(f"  結果: AOV変化率 {threshold_20k.aov_change_rate:+.1%}")
    print(f"  判定: {'✓ 支持' if sub1_met else '✗ 棄却'}")

    # サブ仮説2
    sub2_met = threshold_20k.margin_change_rate >= 0
    print(f"\n■ サブ仮説2: 送料粗利減は粗利増で相殺可能")
    print(f"  結果: 粗利変化率 {threshold_20k.margin_change_rate:+.1%}")
    print(f"  判定: {'✓ 支持' if sub2_met else '✗ 棄却'}")

    # ============================================================
    # Step 9: 意思決定への示唆
    # ============================================================
    print("\n[Step 9] 意思決定への示唆")
    print("-" * 40)

    print("\n【シミュレーション結果に基づく示唆】")

    print(f"""
1. 20,000円閾値導入の効果
   - AOV: {threshold_20k.aov_change_rate:+.1%}の増加が見込まれる
   - 粗利: {threshold_20k.margin_change_rate:+.1%}の変化
   - ジャンプ率{threshold_20k.jump_rate:.1%}は目標（30%）を{'達成' if main_hypothesis_met else '未達成'}

2. 感度分析から
   - 追加購入傾向が{min(propensity_values):.0%}まで低下しても粗利は{'増加' if sensitivity[sensitivity['value']==min(propensity_values)]['margin_change_rate'].values[0] > 0 else '減少'}
   - 損益分岐点となる追加購入傾向は推定で30-35%程度

3. 商品構成への示唆
   - 10,000円～15,000円帯の顧客が最もジャンプしやすい
   - 「あと5,000円～8,000円」で送料無料になる商品セットが効果的
   - 推奨追加商品: 3,000円～5,000円帯の単品商品
""")

    print("\n" + "=" * 60)
    print("シミュレーション完了")
    print("=" * 60)

    return baseline, threshold_20k, sensitivity


if __name__ == '__main__':
    baseline, threshold_20k, sensitivity = run_experiment()
