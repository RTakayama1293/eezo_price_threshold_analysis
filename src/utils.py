"""
EEZO 価格帯ジャンプ効果シミュレーター - ユーティリティ
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def load_json(path: str) -> Dict[str, Any]:
    """JSONファイルを読み込む"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str, indent: int = 2):
    """JSONファイルに保存"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    print(f"JSONを保存しました: {path}")


def format_currency(amount: float, unit: str = '円') -> str:
    """金額をフォーマット"""
    return f'¥{amount:,.0f}'


def format_percentage(rate: float, sign: bool = False) -> str:
    """割合をパーセント表示"""
    if sign:
        return f'{rate:+.1%}'
    return f'{rate:.1%}'


def generate_experiment_id() -> str:
    """実験IDを生成"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'exp_{timestamp}'


def ensure_dir(path: str):
    """ディレクトリが存在しない場合は作成"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """タイムスタンプを取得"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


class ReportGenerator:
    """Markdownレポート生成"""
    
    def __init__(self):
        self.content = []
    
    def add_header(self, text: str, level: int = 1):
        """ヘッダーを追加"""
        self.content.append(f'{"#" * level} {text}\n')
    
    def add_paragraph(self, text: str):
        """段落を追加"""
        self.content.append(f'{text}\n')
    
    def add_table(self, headers: list, rows: list):
        """テーブルを追加"""
        # ヘッダー行
        header_line = '| ' + ' | '.join(headers) + ' |'
        separator = '|' + '|'.join(['---'] * len(headers)) + '|'
        
        self.content.append(header_line)
        self.content.append(separator)
        
        # データ行
        for row in rows:
            row_line = '| ' + ' | '.join(str(cell) for cell in row) + ' |'
            self.content.append(row_line)
        
        self.content.append('')
    
    def add_metric_summary(self, metrics: Dict[str, Any]):
        """指標サマリーを追加"""
        self.add_header('主要指標', 2)
        
        headers = ['指標', 'Before', 'After', '変化']
        rows = []
        
        for name, values in metrics.items():
            rows.append([
                name,
                format_currency(values['before']) if 'before' in values else '-',
                format_currency(values['after']) if 'after' in values else '-',
                format_percentage(values['change'], sign=True) if 'change' in values else '-'
            ])
        
        self.add_table(headers, rows)
    
    def add_figure_reference(self, path: str, caption: str):
        """図への参照を追加"""
        self.content.append(f'![{caption}]({path})\n')
        self.content.append(f'*{caption}*\n')
    
    def generate(self) -> str:
        """レポートを生成"""
        return '\n'.join(self.content)
    
    def save(self, path: str):
        """レポートを保存"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.generate())
        print(f"レポートを保存しました: {path}")


def generate_simulation_report(
    baseline,
    treatment,
    experiment_id: str,
    output_dir: str = 'outputs/reports'
) -> str:
    """
    シミュレーション結果からレポートを生成
    
    Args:
        baseline: ベースライン結果
        treatment: 処置群（閾値シナリオ）結果
        experiment_id: 実験ID
        output_dir: 出力ディレクトリ
    
    Returns:
        str: レポートファイルのパス
    """
    ensure_dir(output_dir)
    
    report = ReportGenerator()
    
    # タイトル
    report.add_header('EEZO 価格帯ジャンプ効果 シミュレーション結果レポート')
    report.add_paragraph(f'**実験ID**: {experiment_id}')
    report.add_paragraph(f'**作成日時**: {datetime.now().strftime("%Y年%m月%d日 %H:%M")}')
    
    # エグゼクティブサマリー
    report.add_header('エグゼクティブサマリー', 2)
    
    jump_rate = treatment.jump_rate
    aov_change = treatment.aov_change_rate
    margin_change = treatment.margin_change_rate
    
    report.add_paragraph(f"""
20,000円送料無料閾値の導入効果:

- **ジャンプ率**: {format_percentage(jump_rate)}（10,000円～20,000円帯の顧客のうち）
- **AOV変化**: {format_percentage(aov_change, sign=True)}
- **粗利変化**: {format_percentage(margin_change, sign=True)}

**結論**: {'導入推奨' if margin_change > 0 else '要検討'}
""")
    
    # 主要指標
    report.add_metric_summary({
        'AOV（平均注文額）': {
            'before': baseline.aov_before,
            'after': treatment.aov_after,
            'change': treatment.aov_change_rate
        },
        '売上総額': {
            'before': baseline.revenue_before,
            'after': treatment.revenue_after,
            'change': treatment.revenue_change_rate
        },
        '粗利総額': {
            'before': baseline.margin_before,
            'after': treatment.margin_after,
            'change': treatment.margin_change_rate
        }
    })
    
    # 仮説と結果
    report.add_header('仮説と検証結果', 2)
    report.add_paragraph("""
**仮説**: 送料無料閾値を20,000円に設定することで、10,000円～15,000円帯の顧客の30%以上が20,000円帯へジャンプする。
""")
    
    hypothesis_result = '支持される' if jump_rate > 0.30 else '支持されない（ジャンプ率不足）'
    report.add_paragraph(f'**結果**: 仮説は **{hypothesis_result}** （ジャンプ率: {format_percentage(jump_rate)}）')
    
    # 示唆
    report.add_header('ビジネスへの示唆', 2)
    
    if margin_change > 0:
        report.add_paragraph("""
1. **送料無料閾値20,000円の導入は収益にプラス**
   - 送料粗利の減少を上回るAOV増効果
   
2. **商品構成の工夫が重要**
   - 8,000円～10,000円の「追加購入しやすい商品」の充実
   - 20,000円セット商品の開発
   
3. **UI/UXでの閾値表示**
   - 「あと○○円で送料無料」の動的表示が効果を最大化
""")
    else:
        report.add_paragraph("""
1. **閾値設定の再検討が必要**
   - 20,000円は高すぎる可能性
   - 15,000円での再シミュレーション推奨
   
2. **または段階的閾値の検討**
   - 10,000円で送料割引、20,000円で無料
""")
    
    # 保存
    timestamp = get_timestamp()
    report_path = f'{output_dir}/{experiment_id}_report_{timestamp}.md'
    report.save(report_path)
    
    return report_path


def main():
    """ユーティリティのテスト"""
    print("=== ユーティリティテスト ===")
    
    # フォーマットテスト
    print(f"金額: {format_currency(12500)}")
    print(f"割合: {format_percentage(0.35)}")
    print(f"変化率: {format_percentage(0.15, sign=True)}")
    
    # タイムスタンプテスト
    print(f"タイムスタンプ: {get_timestamp()}")
    print(f"実験ID: {generate_experiment_id()}")


if __name__ == '__main__':
    main()
