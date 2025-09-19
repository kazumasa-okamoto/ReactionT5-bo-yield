"""
Buchwald-Hartwig 反応の生成スクリプト
- 入力の CSV（列: 'Aryl halide', 'Ligand', 'Base', 'Additive' を想定）を読み込み
- 規定のロジックに従って 'product' 列と 'rxn_smiles' 列を生成
- 出力 CSV に保存
"""

from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions

def canonicalize_with_dict(smi: str, can_smi_dict: Dict[str, str] = {}) -> str:
    if smi not in can_smi_dict:
        # None/NaN 対応: 空文字列として扱う
        if smi is None or (isinstance(smi, float) and pd.isna(smi)):
            can = ""
        else:
            mol = Chem.MolFromSmiles(str(smi))
            can = Chem.MolToSmiles(mol) if mol else ""
        can_smi_dict[smi] = can
        return can
    else:
        return can_smi_dict[smi]


def generate_buchwald_hartwig_rxns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # 元の DataFrame を破壊しないようコピー
    df = df.copy()

    # 反応 SMARTS（提示コードのまま）
    fwd_template = '[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]'
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template)

    # アミン: p-トルイジン（メチルアニリン）
    methylaniline = 'Cc1ccc(N)cc1'
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)

    # Pd 触媒
    pd_catalyst = Chem.MolToSmiles(Chem.MolFromSmiles('O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F'))

    # --- product 列の生成 ---
    products: List[str] = []
    for i, row in df.iterrows():
        # 必要列が欠損していないかチェック（Aryl halide は必須）
        aryl_halide_smi = row.get('Aryl halide', '')
        if pd.isna(aryl_halide_smi):
            aryl_halide_smi = ''
        aryl_halide_mol = Chem.MolFromSmiles(str(aryl_halide_smi)) if aryl_halide_smi else None

        if aryl_halide_mol is None or methylaniline_mol is None:
            products.append('')
            continue

        reacts = (aryl_halide_mol, methylaniline_mol)
        rxn_products = rxn.RunReactants(reacts)

        # 生成物の SMILES（重複排除）
        rxn_products_smiles = set()
        for prods in rxn_products:
            # 反応一発につき tuple で返る。先頭生成物のみ使用
            if len(prods) == 0 or prods[0] is None:
                continue
            smi = Chem.MolToSmiles(prods[0])
            if smi:
                rxn_products_smiles.add(smi)

        # 生成物候補の数に応じて処理
        if len(rxn_products_smiles) == 1:
            products.append(list(rxn_products_smiles)[0])
        elif len(rxn_products_smiles) == 0:
            products.append('')
        else:
            # 複数候補が出た場合は一意に決められないため、ソートして先頭を採用しつつ警告
            products.append(sorted(list(rxn_products_smiles))[0])
            print(f"[WARN] Row {i}: 複数の生成物候補が見つかりました: {rxn_products_smiles}. 先頭を採用します。", file=sys.stderr)

    df['product'] = products

    # 追加: 定数SMILES列（行ごとに同一値を入れる）
    df['methylaniline'] = methylaniline
    df['pd_catalyst'] = pd_catalyst

    # --- 反応 SMILES（reactants >> product）列の生成 ---
    rxns: List[str] = []
    can_smiles_dict: Dict[str, str] = {}

    # 入力列が無い場合のために空文字で補完
    for col in ['Aryl halide', 'Ligand', 'Base', 'Additive']:
        if col not in df.columns:
            df[col] = ''

    for i, row in df.iterrows():
        aryl_halide = canonicalize_with_dict(row['Aryl halide'], can_smiles_dict)
        can_smiles_dict[row['Aryl halide']] = aryl_halide

        ligand = canonicalize_with_dict(row['Ligand'], can_smiles_dict)
        can_smiles_dict[row['Ligand']] = ligand

        base = canonicalize_with_dict(row['Base'], can_smiles_dict)
        can_smiles_dict[row['Base']] = base

        additive = canonicalize_with_dict(row['Additive'], can_smiles_dict)
        can_smiles_dict[row['Additive']] = additive

        reactants = f"{aryl_halide}.{methylaniline}.{pd_catalyst}.{ligand}.{base}.{additive}"
        rxns.append(f"{reactants}>>{row.get('product', '')}")

    df['rxn_smiles'] = rxns
    return df, rxns

def main():
    # 内部でパスを指定
    input_path = Path(__file__).parent.parent.parent / 'data' / "Buchwald-Hartwig" / 'Dreher_and_Doyle_complete_grid.csv'
    output_path = Path(__file__).parent.parent.parent / 'data' / "Buchwald-Hartwig" / 'Dreher_and_Doyle_reaction.csv'

    # CSV 読み込み
    df = pd.read_csv(input_path)

    # 欠損列の警告
    for col in ['Aryl halide', 'Ligand', 'Base', 'Additive']:
        if col not in df.columns:
            print(f"[WARN] 入力 CSV に '{col}' 列が見つかりません。空文字で補完します。", file=sys.stderr)

    # 生成
    out_df, _ = generate_buchwald_hartwig_rxns(df)

    # 出力
    out_df.to_csv(output_path, index=False)
    print(f"[INFO] 書き出し完了: {output_path}")
    
    return out_df


if __name__ == "__main__":
    main()

