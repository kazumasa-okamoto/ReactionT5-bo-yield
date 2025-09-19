"""
Excelの全シートを結合してCSV化し、重複行をまとめて削除するユーティリティ。
- 元シート名を `_source_sheet` 列として付与します。
- 入力条件（Ligand, Additive, Base, Aryl halide）一致での重複除去を行います。
"""

from pathlib import Path
import pandas as pd

def read_all_sheets_as_strings(xlsx_path: Path) -> pd.DataFrame:
    """Excelの全シートを文字列型として読み込み、_source_sheet列を付与して縦結合する。"""
    xls = pd.ExcelFile(xlsx_path)
    frames = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)
        # 空シートはスキップ
        if df.shape[0] == 0 and df.shape[1] == 0:
            continue
        # 先頭に元シート名を追加
        df.insert(0, "_source_sheet", sheet)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    # 列の差異を許容して縦結合（存在しない列はNaN）
    combined = pd.concat(frames, ignore_index=True, sort=False)

    # 文字列統一: 前後の空白を除去（NaNはそのまま）
    combined = combined.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return combined

def agg_join(series: pd.Series) -> str:
    """同一キーで集約する際に、重複を排して ; 区切りで連結する。空やNaNは除外。"""
    uniq = sorted(set([str(x) for x in series if pd.notna(x) and str(x).strip() != ""]))
    return ";".join(uniq) if uniq else ""

def dedup_by_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    入力条件（Ligand, Additive, Base, Aryl halide）が完全一致の行を1つにまとめる。
    - 判定キー: ["Ligand", "Additive", "Base", "Aryl halide"]
    - それ以外の列（たとえば Output, _source_sheet）は ; 区切りで集約
    """
    required = ["Ligand", "Additive", "Base", "Aryl halide"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"必要な列が見つかりません: {col}")

    key_cols = required
    # 集約対象列を決定（キー以外の全列を ; 結合）
    agg_cols = [c for c in df.columns if c not in key_cols]
    agg_dict = {c: agg_join for c in agg_cols}

    result = (df.groupby(key_cols, dropna=False, as_index=False)
                .agg(agg_dict))
    return result

def main():
    xlsx_path = Path(__file__).parent.parent.parent / 'data' / "Buchwald-Hartwig"/ 'Dreher_and_Doyle_input_data.xlsx'
    output = Path(__file__).parent.parent.parent / 'data' / "Buchwald-Hartwig" / 'Dreher_and_Doyle_input_data.csv'

    # 1) 全シート結合
    combined = read_all_sheets_as_strings(xlsx_path)

    # 2) 入力条件一致で重複除去
    dedup_inputs = dedup_by_inputs(combined)
    dedup_inputs.to_csv(output, index=False, encoding="utf-8-sig")

    # 処理サマリを表示
    print("=== 処理サマリ ===")
    print(f"入力Excel: {xlsx_path}")
    print(f"シート結合: {len(combined)} 行, {combined.shape[1]} 列")
    print(f"重複除去(入力条件一致): {len(dedup_inputs)} 行 -> {output}")

if __name__ == "__main__":
    main()

