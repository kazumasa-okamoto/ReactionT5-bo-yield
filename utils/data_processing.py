import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger 

# RDKitのエラーメッセージを非表示にして出力をクリーンに保つ
RDLogger.DisableLog('rdApp.*')

def convert_inchi_to_smiles(inchi_string: str) -> str:
    """
    InChI文字列をSMILES文字列に変換する。
    変換に失敗した場合は空文字を返す。
    """
    if pd.isna(inchi_string) or not isinstance(inchi_string, str):
        return ''
    mol = Chem.MolFromInchi(inchi_string)
    if mol:
        return Chem.MolToSmiles(mol)
    return ''

# --- STEP 1: ファイルと定数の定義 ---
INPUT_FILENAME = Path(__file__).parent.parent / 'data' / 'inchi_23l.csv'
OUTPUT_FILENAME = Path(__file__).parent.parent / 'data' / 'inchi_23l_reaction_t5_ready.csv'

# 全ての反応で共通の求核剤（Nucleophile）
NUCLEOPHILE_SMILES = 'OB(O)B(O)O' # テトラヒドロキシジボロン

print(f"処理を開始します: '{INPUT_FILENAME}' -> '{OUTPUT_FILENAME}'")

# --- STEP 2: データの読み込み ---
try:
    df = pd.read_csv(INPUT_FILENAME)
    print(f"'{INPUT_FILENAME}'を正常に読み込みました。データ数: {len(df)}件")
except FileNotFoundError:
    print(f"[エラー] ファイルが見つかりません: '{INPUT_FILENAME}'")
    exit()

# --- STEP 3: InChIからSMILESへの変換 ---
print("化学構造をInChI形式からSMILES形式に変換中...")

# 変化する要素を変換
df['electrophile_smiles'] = df['electrophile_inchi'].apply(convert_inchi_to_smiles)
df['ligand_smiles'] = df['ligand_inchi'].apply(convert_inchi_to_smiles)
df['solvent_smiles'] = df['solvent_inchi'].apply(convert_inchi_to_smiles)
df['product_smiles'] = df['product_inchi'].apply(convert_inchi_to_smiles)

print("SMILESへの変換が完了しました。")

# --- STEP 4: Reaction T5のフォーマットに従って列を構築 ---
print("Reaction T5の入力形式に合わせてデータを整形中...")

# 新しいDataFrameを作成
t5_df = pd.DataFrame()

df['nucleophile_smiles'] = NUCLEOPHILE_SMILES

# REACTANT列: 求電子剤
t5_df['REACTANT'] = df['electrophile_smiles'] 

# REAGENT列:求核材（ボロン酸）
t5_df['REAGENT'] = df['nucleophile_smiles']

# "CATALYST"列: リガンド
t5_df['CATALYST'] = df['ligand_smiles']

# "SOLVENT"列: 溶媒
t5_df['SOLVENT'] = df['solvent_smiles']
# PRODUCT列
t5_df['PRODUCT'] = df['product_smiles']

# YIELD列
t5_df['YIELD'] = df['yield']

print("データの整形が完了しました。")

# --- STEP 5: 結果を新しいCSVファイルに保存 ---
t5_df.to_csv(OUTPUT_FILENAME, index=False)

print("-" * 50)
print(f"🎉 処理がすべて完了しました！")
print(f"出力ファイル: '{OUTPUT_FILENAME}'")
print("\n生成されたデータの先頭5行:")
print(t5_df.head())
print("-" * 50)

