"""
EEZO 価格帯ジャンプ効果シミュレーター - 可視化モジュール
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

# 日本語フォント設定
try:
    import japanize_matplotlib
except ImportError:
    print("警告: japanize_matplotlib がインストールされていません。")
    print("pip install japanize-matplotlib でインストールしてください。")

# スタイル設定
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


class Visualizer:
    """シミュレーション結果の可視化"""
    
    def __init__(self, figsize: tuple = (10, 6), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_distribution_shift(
        self,
        baseline,
        treatment,
        title: str = '価格帯分布の変化',
        save_path: Optional[str] = None
    ):
        """
        価格帯分布の変化を可視化（Before/After比較）
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        tier_labels = {
            'tier_0_5000': '～5,000円',
            'tier_5000_10000': '5,000～10,000円',
            'tier_10000_15000': '10,000～15,000円',
            'tier_15000_20000': '15,000～20,000円',
            'tier_20000_plus': '20,000円以上'
        }
        
        tier_order = list(tier_labels.keys())
        
        # Before（ベースライン）
        before_data = []
        for tier in tier_order:
            before_data.append(baseline.before_distribution.get(tier, 0))
        
        # After（閾値シナリオ）
        after_data = []
        for tier in tier_order:
            after_data.append(treatment.after_distribution.get(tier, 0))
        
        x = np.arange(len(tier_order))
        width = 0.35
        
        # 左: 分布比較
        axes[0].bar(x - width/2, before_data, width, label='Before（閾値なし）', color='#3498db')
        axes[0].bar(x + width/2, after_data, width, label='After（20K閾値）', color='#e74c3c')
        
        axes[0].set_ylabel('顧客構成比')
        axes[0].set_title('価格帯別顧客分布')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([tier_labels[t] for t in tier_order], rotation=45, ha='right')
        axes[0].legend()
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # 右: 変化量
        change = [a - b for a, b in zip(after_data, before_data)]
        colors = ['#27ae60' if c > 0 else '#e74c3c' for c in change]
        
        axes[1].bar(x, change, color=colors)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_ylabel('変化量')
        axes[1].set_title('価格帯別変化（After - Before）')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([tier_labels[t] for t in tier_order], rotation=45, ha='right')
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.0%}'))
        
        # ジャンプ効果のアノテーション
        # 10,000-20,000円帯の減少と20,000円以上の増加を強調
        for i, tier in enumerate(tier_order):
            if tier in ['tier_10000_15000', 'tier_15000_20000'] and change[i] < 0:
                axes[1].annotate('ジャンプ元', (i, change[i]), 
                               textcoords="offset points", xytext=(0, -15),
                               ha='center', fontsize=8, color='#e74c3c')
            elif tier == 'tier_20000_plus' and change[i] > 0:
                axes[1].annotate('ジャンプ先', (i, change[i]),
                               textcoords="offset points", xytext=(0, 10),
                               ha='center', fontsize=8, color='#27ae60')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"図を保存しました: {save_path}")
        
        plt.show()
        return fig
    
    def plot_revenue_impact(
        self,
        baseline,
        treatment,
        title: str = '収益インパクト',
        save_path: Optional[str] = None
    ):
        """
        収益インパクトを可視化
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        metrics = ['AOV', '売上総額', '粗利総額']
        before_values = [
            baseline.aov_before,
            baseline.revenue_before,
            baseline.margin_before
        ]
        after_values = [
            treatment.aov_after,
            treatment.revenue_after,
            treatment.margin_after
        ]
        changes = [
            treatment.aov_change_rate,
            treatment.revenue_change_rate,
            treatment.margin_change_rate
        ]
        
        for i, (metric, before, after, change) in enumerate(zip(metrics, before_values, after_values, changes)):
            ax = axes[i]
            
            x = ['Before', 'After']
            y = [before, after]
            colors = ['#3498db', '#27ae60' if change > 0 else '#e74c3c']
            
            bars = ax.bar(x, y, color=colors)
            
            # 値をバーの上に表示
            for bar, val in zip(bars, y):
                height = bar.get_height()
                ax.annotate(f'¥{val:,.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
            
            # 変化率を表示
            ax.set_title(f'{metric}\n({change:+.1%})', fontsize=12, fontweight='bold')
            ax.set_ylabel('金額（円）')
            
            # y軸のフォーマット
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'¥{y/1000:.0f}K'))
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"図を保存しました: {save_path}")
        
        plt.show()
        return fig
    
    def plot_jump_analysis(
        self,
        result,
        title: str = 'ジャンプ効果の詳細分析',
        save_path: Optional[str] = None
    ):
        """
        ジャンプ効果の詳細分析を可視化
        """
        df = result.customer_data
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. ジャンプ確率 vs 元の金額
        ax1 = axes[0, 0]
        jump_candidates = df[(df['original_amount'] >= 5000) & (df['original_amount'] < 20000)]
        ax1.scatter(jump_candidates['original_amount'], 
                   jump_candidates['jump_probability'],
                   alpha=0.5, s=20)
        ax1.axhline(y=0.58, color='red', linestyle='--', label='基本ジャンプ傾向 (58%)')
        ax1.axvline(x=10000, color='green', linestyle='--', alpha=0.5, label='10,000円ライン')
        ax1.axvline(x=15000, color='green', linestyle='--', alpha=0.5, label='15,000円ライン')
        ax1.set_xlabel('元の注文金額（円）')
        ax1.set_ylabel('ジャンプ確率')
        ax1.set_title('元の金額とジャンプ確率の関係')
        ax1.legend()
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'¥{x/1000:.0f}K'))
        
        # 2. ジャンプ前後の金額分布
        ax2 = axes[0, 1]
        jumped = df[df['jump_decision'] == True]
        if len(jumped) > 0:
            ax2.hist(jumped['original_amount'], bins=20, alpha=0.5, label='ジャンプ前', color='#3498db')
            ax2.hist(jumped['after_amount'], bins=20, alpha=0.5, label='ジャンプ後', color='#e74c3c')
            ax2.axvline(x=20000, color='green', linestyle='--', label='閾値 (20,000円)')
            ax2.set_xlabel('注文金額（円）')
            ax2.set_ylabel('顧客数')
            ax2.set_title('ジャンプした顧客の金額変化')
            ax2.legend()
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'¥{x/1000:.0f}K'))
        
        # 3. 価格帯別ジャンプ率
        ax3 = axes[1, 0]
        tier_jump_rates = df.groupby('original_tier')['jump_decision'].mean()
        tier_labels = {
            'tier_0_5000': '～5K',
            'tier_5000_10000': '5K～10K',
            'tier_10000_15000': '10K～15K',
            'tier_15000_20000': '15K～20K',
            'tier_20000_plus': '20K～'
        }
        tier_order = ['tier_5000_10000', 'tier_10000_15000', 'tier_15000_20000']
        
        x = [tier_labels.get(t, t) for t in tier_order]
        y = [tier_jump_rates.get(t, 0) for t in tier_order]
        colors = ['#f39c12' if r < 0.3 else '#27ae60' if r > 0.3 else '#3498db' for r in y]
        
        bars = ax3.bar(x, y, color=colors)
        ax3.set_xlabel('価格帯')
        ax3.set_ylabel('ジャンプ率')
        ax3.set_title('価格帯別ジャンプ率')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # 値をバーの上に表示
        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax3.annotate(f'{val:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # 4. ジャンプによる増加金額の分布
        ax4 = axes[1, 1]
        if len(jumped) > 0:
            jumped['amount_increase'] = jumped['after_amount'] - jumped['original_amount']
            ax4.hist(jumped['amount_increase'], bins=20, color='#27ae60', alpha=0.7)
            mean_increase = jumped['amount_increase'].mean()
            ax4.axvline(x=mean_increase, color='red', linestyle='--', 
                       label=f'平均増加額: ¥{mean_increase:,.0f}')
            ax4.set_xlabel('増加金額（円）')
            ax4.set_ylabel('顧客数')
            ax4.set_title('ジャンプによる注文金額の増加')
            ax4.legend()
            ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'¥{x/1000:.0f}K'))
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"図を保存しました: {save_path}")
        
        plt.show()
        return fig
    
    def plot_sensitivity(
        self,
        sensitivity_df: pd.DataFrame,
        target_metric: str = 'margin_change_rate',
        title: str = '感度分析',
        save_path: Optional[str] = None
    ):
        """
        感度分析結果を可視化
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        metric_labels = {
            'jump_rate': 'ジャンプ率',
            'aov_change_rate': 'AOV変化率',
            'revenue_change_rate': '売上変化率',
            'margin_change_rate': '粗利変化率'
        }
        
        ax.plot(sensitivity_df['value'], sensitivity_df[target_metric], 
               marker='o', linewidth=2, markersize=8)
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='損益分岐点')
        ax.fill_between(sensitivity_df['value'], 0, sensitivity_df[target_metric],
                       where=(sensitivity_df[target_metric] > 0),
                       alpha=0.3, color='green', label='利益増')
        ax.fill_between(sensitivity_df['value'], 0, sensitivity_df[target_metric],
                       where=(sensitivity_df[target_metric] < 0),
                       alpha=0.3, color='red', label='利益減')
        
        ax.set_xlabel(f'パラメータ値: {sensitivity_df["parameter"].iloc[0]}')
        ax.set_ylabel(metric_labels.get(target_metric, target_metric))
        ax.set_title(f'{title}: {sensitivity_df["parameter"].iloc[0]}')
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.1%}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"図を保存しました: {save_path}")
        
        plt.show()
        return fig


def main():
    """可視化モジュールのテスト"""
    from simulator import ThresholdJumpSimulator
    
    # シミュレーション実行
    sim = ThresholdJumpSimulator(seed=42)
    
    baseline = sim.run(scenario_name='baseline', threshold=None, n_customers=1000)
    threshold_20k = sim.run(scenario_name='threshold_20k', threshold=20000, n_customers=1000)
    
    # 可視化
    viz = Visualizer()
    
    print("1. 価格帯分布の変化")
    viz.plot_distribution_shift(baseline, threshold_20k)
    
    print("\n2. 収益インパクト")
    viz.plot_revenue_impact(baseline, threshold_20k)
    
    print("\n3. ジャンプ効果の詳細分析")
    viz.plot_jump_analysis(threshold_20k)


if __name__ == '__main__':
    main()
