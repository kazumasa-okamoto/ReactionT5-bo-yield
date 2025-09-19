import pandas as pd
from pathlib import Path

# --- STEP 1: ファイルと定数の定義 ---
INPUT_FILENAME = Path(__file__).parent.parent.parent / 'data' / "Buchwald-Hartwig" / 'Dreher_and_Doyle_reaction.csv'
OUTPUT_FILENAME = Path(__file__).parent.parent.parent / 'data' / "Buchwald-Hartwig" / 'Dreher_and_Doyle_reaction_t5_ready.csv'

print(f"処理を開始します: '{INPUT_FILENAME}' -> '{OUTPUT_FILENAME}'")

# --- STEP 2: データの読み込み ---
try:
    df = pd.read_csv(INPUT_FILENAME)
    print(f"'{INPUT_FILENAME}'を正常に読み込みました。データ数: {len(df)}件")
except FileNotFoundError:
    print(f"[エラー] ファイルが見つかりません: '{INPUT_FILENAME}'")
    exit()

# --- STEP 3: Reaction T5のフォーマットに従って列を構築 ---
print("Reaction T5の入力形式に合わせてデータを整形中...")

# 新しいDataFrameを作成
t5_df = pd.DataFrame()

# REACTANT列: ハロゲン化アリール + メチルアミン
t5_df['REACTANT'] = df['Aryl halide'].astype(str) + "." + df['methylaniline'].astype(str)

# REAGENT列: 塩基 + 添加剤
t5_df['REAGENT'] = df['Base'].astype(str) + "." + df['Additive'].astype(str)

# CATALYST列: Pd触媒 + リガンド
t5_df['CATALYST'] = df['pd_catalyst'].astype(str) + "." + df['Ligand'].astype(str)

# SOLVENT列: 情報なし
t5_df['SOLVENT'] = ""

# PRODUCT列: 生成物
t5_df['PRODUCT'] = df['product']

# YIELD列: 収率
t5_df['YIELD'] = df['Output']

print("データの整形が完了しました。")

# --- STEP 4: 欠損値のチェックと処理 ---
print("欠損値をチェック中...")

# 欠損値がある行を特定
missing_mask = t5_df.isnull().any(axis=1)
missing_count = missing_mask.sum()

if missing_count > 0:
    print(f"警告: {missing_count}件の行に欠損値が含まれています。")
    print("欠損値を含む行を除外します...")
    t5_df = t5_df.dropna()
    print(f"欠損値除去後のデータ数: {len(t5_df)}件")
else:
    print("欠損値は見つかりませんでした。")

# --- STEP 5: 結果を新しいCSVファイルに保存 ---
t5_df.to_csv(OUTPUT_FILENAME, index=False)

print("-" * 50)
print(f"🎉 処理がすべて完了しました！")
print(f"出力ファイル: '{OUTPUT_FILENAME}'")
print("\n生成されたデータの先頭5行:")
print(t5_df.head())
print(f"\n最終データ数: {len(t5_df)}件")
print("-" * 50)

