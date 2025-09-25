"""
最大の完全格子を4要因（Ligand, Additive, Base, Aryl halide）の中で選び、
組合せ数（|L|×|A|×|B|×|AH|）が最大となる構成を探索して出力するスクリプト。

- LigandとBaseをアンカーとして探索を実施し、最大の完全格子を採用する。
- 出力：
  - 最大完全格子に該当する行のみの CSV（重複除去済み）
  - 積の分解表示（例: 2 × 8 × 5 × 3 = 240）
  - 完全格子化のために除外された水準一覧
  - アンカーペア（LigandとBase）
"""

from pathlib import Path
from itertools import combinations, product
import pandas as pd

# 対象カラム（4要因）
FACTOR_COLS = [
    "Ligand",
    "Additive",
    "Base",
    "Aryl halide",
]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """前処理：対象4列を文字列化し前後空白を除去。"""
    df = df.copy()
    for c in FACTOR_COLS:
        if c in df.columns:
            df[c] = df[c].astype(object)
            mask = df[c].notna()
            df.loc[mask, c] = df.loc[mask, c].astype(str).str.strip()
    return df


def unique_levels(df: pd.DataFrame):
    """各要因のユニーク水準を取得（Noneは除外）。"""
    levels = {}
    for c in FACTOR_COLS:
        arr = df[c].apply(lambda x: None if pd.isna(x) else str(x).strip())
        uniq = pd.unique(arr).tolist()
        # Noneを除外
        uniq = [x for x in uniq if x is not None]
        levels[c] = uniq
    return levels


def observed_4tuples(df: pd.DataFrame) -> set:
    """観測された4要因の完全タプル集合。"""
    obs = set()
    for _, row in df.iterrows():
        tup = tuple(
            (None if pd.isna(row[c]) else str(row[c]).strip())
            for c in FACTOR_COLS
        )
        obs.add(tup)
    return obs


def build_all_pair_maps(df: pd.DataFrame):
    """
    4要因の全てのペア (i, j), i<j についてマップを構築。
    """
    df = normalize(df)
    n = len(FACTOR_COLS)
    pair_maps = {}

    for i in range(n):
        for j in range(i + 1, n):
            coli, colj = FACTOR_COLS[i], FACTOR_COLS[j]
            pairs_present = set()
            sets_by_pair = {k: {} for k in range(n) if k not in (i, j)}

            for (ai_raw, aj_raw), g in df.groupby([coli, colj], dropna=False):
                ai = None if pd.isna(ai_raw) else str(ai_raw).strip()
                aj = None if pd.isna(aj_raw) else str(aj_raw).strip()
                pairs_present.add((ai, aj))

                for k in sets_by_pair.keys():
                    colk = FACTOR_COLS[k]
                    vals = (
                        g[colk]
                        .apply(lambda v: None if pd.isna(v) else str(v).strip())
                        .unique()
                        .tolist()
                    )
                    sets_by_pair[k][(ai, aj)] = set(vals)

            pair_maps[(i, j)] = {
                "pairs_present": pairs_present,
                "sets_by_pair": sets_by_pair,
            }

    return pair_maps


def all_nonempty_subsets(lst):
    """非空部分集合を全列挙（サイズ1〜|lst|）。"""
    n = len(lst)
    for k in range(1, n + 1):
        for comb in combinations(lst, k):
            yield list(comb)


def is_full_grid(retained_sets: dict, observed_set: set) -> bool:
    """
    retained_sets = {col_name: set(...)} が与えられたとき、
    4要因の直積の全組合せが observed_set（観測済み4タプル）に
    すべて含まれているかを厳密にチェックする。
    """
    iters = [retained_sets[c] for c in FACTOR_COLS]
    for tup in product(*iters):
        if tup not in observed_set:
            return False
    return True


def eval_product_for_pair(i, j, A_set, B_set, levels_all_by_name, pair_map, observed_set):
    """
    アンカー要因 i, j の候補集合 A_set, B_set が与えられたとき、
    すべてのクロスペア (a,b) ∈ A_set×B_set で観測されている残り2要因の共通集合を取り、
    さらに"4要因全直積の完全被覆"を厳密チェックして合格した場合のみ、
    総直積サイズと水準集合を返す。

    返り値:
      - prod: 直積サイズ（完全格子でなければ 0）
      - retained_sets: { col_name: set(...) } の辞書（完全格子の場合のみ）
    """
    cross_pairs = set(product(A_set, B_set))
    if not cross_pairs.issubset(pair_map["pairs_present"]):
        return (0, None)

    retained_sets = {}
    n = len(FACTOR_COLS)

    retained_sets[FACTOR_COLS[i]] = set(A_set)
    retained_sets[FACTOR_COLS[j]] = set(B_set)

    # 残り2要因の交差をとる
    for k in range(n):
        if k in (i, j):
            continue
        colk = FACTOR_COLS[k]
        inter = set(levels_all_by_name[colk])
        kmap = pair_map["sets_by_pair"][k]
        for p in cross_pairs:
            inter &= kmap.get(p, set())
            if not inter:
                return (0, None)
        retained_sets[colk] = inter

    # ここで"4要因全直積が観測済みか"を厳密チェック
    if not is_full_grid(retained_sets, observed_set):
        return (0, None)

    # 直積サイズ
    prod = 1
    for col in FACTOR_COLS:
        prod *= len(retained_sets[col])

    return (prod, retained_sets)


def find_max_complete_grid(df: pd.DataFrame):
    """
    LigandとBaseをアンカーとして完全格子を探索し、最大の完全格子を返す。
    戻り値: dict（構成・積・採用アンカーペア）
    """
    df = normalize(df)
    levels = unique_levels(df)
    levels_all_by_name = {c: levels[c] for c in FACTOR_COLS}

    # 観測された4タプル（完全キー）の集合
    observed_set = observed_4tuples(df)

    # 全ペアのマップを構築
    pair_maps = build_all_pair_maps(df)

    best = {"prod": 0}

    # LigandとBaseのみをアンカーとして探索
    i, j = 0, 2  # Ligandは0番目、Baseは2番目
    coli, colj = FACTOR_COLS[i], FACTOR_COLS[j]
    Ai_all = levels_all_by_name[coli]
    Bj_all = levels_all_by_name[colj]

    for A_set in all_nonempty_subsets(Ai_all):
        for B_set in all_nonempty_subsets(Bj_all):
            prod, retained = eval_product_for_pair(
                i, j, A_set, B_set, levels_all_by_name, pair_maps[(i, j)], observed_set
            )
            if prod > best["prod"]:
                best = {
                    "prod": prod,
                    "anchor_pair": (coli, colj),
                    **{col: list(retained[col]) for col in FACTOR_COLS},
                }

    return best


def extract_full_grid_rows(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    見つけた最大完全格子（config）の行のみを抽出し、重複を除去（Noneは除外）。
    """
    df = normalize(df)
    mask = pd.Series(True, index=df.index)

    for col in FACTOR_COLS:
        vals = config[col]
        # Noneは除外して処理
        vals_wo_none = [v for v in vals if v is not None]

        if vals_wo_none:
            cond = df[col].apply(lambda x: (not pd.isna(x)) and (str(x).strip() in vals_wo_none))
        else:
            cond = pd.Series(False, index=df.index)

        mask = mask & cond

    sub = df.loc[mask].drop_duplicates()
    return sub

def _product_breakdown(best: dict) -> str:
    """積の分解を 'a × b × c × d = N' の形式で返す。"""
    counts = [
        len(best["Ligand"]),
        len(best["Additive"]),
        len(best["Base"]),
        len(best["Aryl halide"]),
    ]
    return " × ".join(str(x) for x in counts) + f" = {best['prod']}"


def main():
    # 入出力パス（必要に応じて変更してください）
    input_path = Path(__file__).parent.parent.parent / "data" / "Buchwald-Hartwig" / "Dreher_and_Doyle_input_data.csv"
    output_csv = Path(__file__).parent.parent.parent / "data" / "Buchwald-Hartwig" / "Dreher_and_Doyle_complete_grid.csv"

    # 入力読込
    df = pd.read_csv(input_path)

    # 全ペア探索 → 最大完全格子（厳密検証込み）
    best = find_max_complete_grid(df)

    # 最大完全格子の行のみを抽出してCSV出力
    sub = extract_full_grid_rows(df, best)
    sub.to_csv(output_csv, index=False)

    # 全水準（探索に用いた全集合）- 表示用にはNoneも含める
    levels_all = {}
    df_norm = normalize(df)
    for c in FACTOR_COLS:
        arr = df_norm[c].apply(lambda x: None if pd.isna(x) else str(x).strip())
        uniq = pd.unique(arr).tolist()
        levels_all[c] = uniq

    # 出力
    print("==== 最大完全格子をCSVに保存しました ====")
    print(f"Path: {output_csv}\n")

    print("==== 完全格子 構成（LigandとBaseをアンカー） ====")
    print(f"アンカーペア: {best.get('anchor_pair')}")
    print(f"L: {len(best['Ligand'])}, Add: {len(best['Additive'])}, "
          f"B: {len(best['Base'])}, AH: {len(best['Aryl halide'])}")
    print(_product_breakdown(best))
    print()

    print("==== 完全格子のために除外された水準（全集合との差） ====")
    print("Lig:", set(levels_all["Ligand"]) - set(best["Ligand"]))
    print("Add:", set(levels_all["Additive"]) - set(best["Additive"]))
    print("Base:", set(levels_all["Base"]) - set(best["Base"]))
    print("Aryl halide:", set(levels_all["Aryl halide"]) - set(best["Aryl halide"]))


if __name__ == "__main__":
    main()

