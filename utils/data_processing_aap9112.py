import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger

# ---- RDKit の冗長ログを抑制 ----
RDLogger.DisableLog('rdApp.*')

# ---- 入出力 ----
INPUT_CSV = Path(__file__).parent.parent / "data" / "aap9112_Data_File_S1.csv"
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "aap9112_reaction_t5_ready.csv"

# ---- 名前→SMILES 対応 ----
reactant_1_smiles = {
    '6-chloroquinoline': 'C1=C(Cl)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC',
    '6-Bromoquinoline': 'C1=C(Br)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC',
    '6-triflatequinoline': 'C1C2C(=NC=CC=2)C=CC=1OS(C(F)(F)F)(=O)=O.CCC1=CC(=CC=C1)CC',
    '6-Iodoquinoline': 'C1=C(I)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC',
    '6-quinoline-boronic acid hydrochloride': 'C1C(B(O)O)=CC=C2N=CC=CC=12.Cl.O',
    'Potassium quinoline-6-trifluoroborate': '[B-](C1=CC2=C(C=C1)N=CC=C2)(F)(F)F.[K+].O',
    '6-Quinolineboronic acid pinacol ester': 'B1(OC(C(O1)(C)C)(C)C)C2=CC3=C(C=C2)N=CC=C3.O',
}

reactant_2_smiles = {
    '2a, Boronic Acid': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B(O)O',
    '2b, Boronic Ester': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B4OC(C)(C)C(C)(C)O4',
    '2c, Trifluoroborate': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1[B-](F)(F)F.[K+]',
    '2d, Bromide': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1Br',
}

catalyst_smiles = {
    'Pd(OAc)2': 'CC(=O)O.CC(=O)O.[Pd]',
}

ligand_smiles = {
    'P(tBu)3': 'CC(C)(C)P(C(C)(C)C)C(C)(C)C',
    'P(Ph)3 ': 'c3c(P(c1ccccc1)c2ccccc2)cccc3',
    'AmPhos': 'CC(C)(C)P(C1=CC=C(C=C1)N(C)C)C(C)(C)C',
    'P(Cy)3': 'C1(CCCCC1)P(C2CCCCC2)C3CCCCC3',
    'P(o-Tol)3': 'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C',
    'CataCXium A': 'CCCCP(C12CC3CC(C1)CC(C3)C2)C45CC6CC(C4)CC(C6)C5',
    'SPhos': 'COc1cccc(c1c2ccccc2P(C3CCCCC3)C4CCCCC4)OC',
    'dtbpf': 'CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.[Fe]',
    'XPhos': 'P(c2ccccc2c1c(cc(cc1C(C)C)C(C)C)C(C)C)(C3CCCCC3)C4CCCCC4',
    'dppf': 'C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.[Fe+2]',
    'Xantphos': 'O6c1c(cccc1P(c2ccccc2)c3ccccc3)C(c7cccc(P(c4ccccc4)c5ccccc5)c67)(C)C',
    'None': '',
}

reagent_1_smiles = {
    'NaOH': '[OH-].[Na+]',
    'NaHCO3': '[Na+].OC([O-])=O',
    'CsF': '[F-].[Cs+]',
    'K3PO4': '[K+].[K+].[K+].[O-]P([O-])([O-])=O',
    'KOH': '[K+].[OH-]',
    'LiOtBu': '[Li+].[O-]C(C)(C)C',
    'Et3N': 'CCN(CC)CC',
    'None': '',
}

solvent_1_smiles = {
    'MeCN': 'CC#N.O',
    'THF': 'C1CCOC1.O',
    'DMF': 'CN(C)C=O.O',
    'MeOH': 'CO.O',
    'MeOH/H2O_V2 9:1': 'CO.O',
    'THF_V2': 'C1CCOC1.O',
}

# 固定の生成物
PRODUCT_SMILES = 'C1=C(C2=C(C)C=CC3N(C4OCCCC4)N=CC2=3)C=CC2=NC=CC=C12'

# ---- ユーティリティ ----
def canon(smiles: str) -> str:
    """RDKit で正規化。失敗時は元の文字列を返す。"""
    s = (smiles or "").replace("~", ".").strip()
    if not s:
        return ""
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else s

def join_nonempty(parts):
    parts = [p for p in parts if p]
    return ".".join(parts) if parts else ""

# ---- メイン処理 ----
def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # 期待列名（aap9112 配布に合わせる）
    need_cols = [
        'Reactant_1_Name', 'Reactant_2_Name',
        'Catalyst_1_Short_Hand', 'Ligand_Short_Hand',
        'Reagent_1_Short_Hand', 'Solvent_1_Short_Hand',
        'Product_Yield_PCT_Area_UV',
    ]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise KeyError(f"想定列が見つかりません: {missing}")

    # 各列を SMILES にマップ（未知値は空にする）
    rct1 = df['Reactant_1_Name'].map(lambda k: canon(reactant_1_smiles.get(k, ''))).fillna('')
    rct2 = df['Reactant_2_Name'].map(lambda k: canon(reactant_2_smiles.get(k, ''))).fillna('')
    pdsrc = df['Catalyst_1_Short_Hand'].map(lambda k: canon(catalyst_smiles.get(k, ''))).fillna('')
    lig   = df['Ligand_Short_Hand'].map(lambda k: canon(ligand_smiles.get(k, ''))).fillna('')
    base  = df['Reagent_1_Short_Hand'].map(lambda k: canon(reagent_1_smiles.get(k, ''))).fillna('')
    solv  = df['Solvent_1_Short_Hand'].map(lambda k: canon(solvent_1_smiles.get(k, ''))).fillna('')
    prod  = pd.Series([canon(PRODUCT_SMILES)] * len(df), index=df.index)

    def dot_join(a: str, b: str) -> str:
        """2要素を'.'で連結（どちらか空なら非空のみ）"""
        if a and b:
            return canon(f"{a}.{b}")
        return a or b

    # REACTANT は Reactant_1 のみ
    reactant_col = rct1

    # REAGENT は Reactant_2 + Base
    reagent_col  = pd.Series([dot_join(b, c) for b, c in zip(rct2, base)], index=df.index)

    # Pd(OAc)2 + 配位子
    catalyst_col = pd.Series([dot_join(a, b) for a, b in zip(pdsrc, lig)], index=df.index)

    t5 = pd.DataFrame({
        'REACTANT': reactant_col,
        'REAGENT' : reagent_col,
        'CATALYST': catalyst_col,
        'SOLVENT' : solv,
        'PRODUCT' : prod,
        'YIELD'   : pd.to_numeric(df['Product_Yield_PCT_Area_UV'], errors='coerce'),
    })

    # 出力
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    t5.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"[DONE] {OUTPUT_CSV}  rows={len(t5)}")

    # 足りない辞書の警告（空SMILESの件数）
    warn_cols = ['REACTANT', 'REAGENT', 'CATALYST', 'SOLVENT', 'PRODUCT']
    empties = {c: int((t5[c].astype(str).str.len() == 0).sum()) for c in warn_cols}
    print("[INFO] empty SMILES counts:", empties)

if __name__ == "__main__":
    main()

