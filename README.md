# ReactionT5v2を用いた化学反応収率のベイズ最適化

## 概要

本プロジェクトでは、事前学習済みTransformerモデル[ReactionT5v2](https://github.com/sagawatatsuya/ReactionT5v2)を用いたベイズ最適化により、化学反応の最適な反応条件を探索します。MC Dropoutにより予測の不確実性を推定し、獲得関数（Upper Confidence Bound）を用いて効率的な探索を実現しています。

### 対象データセット

- **NiB**: Nickel-catalyzed Borylation（[ochem-data](https://github.com/doyle-lab-ucla/ochem-data/tree/main/NiB)）
- **Suzuki-Miyaura (SM)**: Suzuki-Miyaura coupling（[rxn_yields](https://github.com/rxn4chemistry/rxn_yields/tree/master/data/Suzuki-Miyaura)）
- **Buchwald-Hartwig (BH)**: Buchwald-Hartwig amination（[rxn_yields](https://github.com/rxn4chemistry/rxn_yields/tree/master/data/Buchwald-Hartwig)）

### 比較手法

- **ReactionT5 BO**: ReactionT5v2 + MC Dropout + ベイズ最適化（提案手法）
- **GPR BO**: Morgan Fingerprint + Gaussian Process Regression + ベイズ最適化（ベースライン）
- **Optuna TPE**: Tree-structured Parzen Estimator（ベースライン）

## セットアップ

### 必要要件

- Python 3.11以上
- CUDA対応GPU（推奨）

### インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd ReactionT5-bo-yield

# 依存パッケージのインストール（uvを使用）
uv sync

# または、pipを使用
pip install -e .
```

## プロジェクト構成

```
ReactionT5-bo-yield/
├── notebooks/               # 分析用Jupyterノートブック
│   ├── check_data.ipynb    # データの類似性確認
│   ├── bo_yield_*.ipynb    # ReactionT5 BOの実験
│   ├── greedy_yield_*.ipynb # ReactionT5 貪欲法の実験
│   ├── optuna_yield_*.ipynb # Optuna TPEの実験
│   ├── gpr_yield_*.ipynb   # GPR BOの実験
│   └── compare_all_results.ipynb # 全結果の比較・可視化
├── scripts/                 # 実験用スクリプト
│   ├── bo_yield/           # ReactionT5 BOの実装
│   ├── gpr/                # GPR BOの実装
│   └── optuna_tpe/         # Optuna TPEの実装
├── data/                    # データセット
│   ├── NiB/
│   ├── Suzuki-Miyaura/
│   ├── Buchwald-Hartwig/
│   └── ORD/
├── data_utils/             # データ前処理スクリプト
├── runs/                   # 実験結果の保存先
└── pyproject.toml          # プロジェクト設定
```

## 実験方法

### 1. データ類似性の確認

```bash
# ノートブックを起動
jupyter notebook notebooks/check_data.ipynb
```

ReactionT5v2の学習データ（ORD）に、各評価データセットと類似の反応が含まれているか確認します。

### 2. ベイズ最適化の実行

各手法について、シード1〜5で複数回実行して統計的評価を行います。

#### ReactionT5 BO（提案手法）

```bash
# スクリプトで実行
python scripts/bo_yield/run_experiment.py --dataset NiB --seed 1

# またはノートブックで実行
jupyter notebook notebooks/bo_yield_NiB.ipynb
```

**特徴:**
- MC Dropout（n_forward=30）による予測の不確実性推定
- Upper Confidence Bound (UCB)獲得関数による探索・活用のバランス
- 10ラウンド × 10試行 = 100試行の最適化

#### GPR BO（ベースライン）

```bash
python scripts/gpr/run_experiment.py --dataset NiB --seed 1
```

**特徴:**
- Morgan Fingerprint（radius=2, nBits=2048）による分子表現
- Gaussian Process Regressionによる予測
- 100試行のベイズ最適化

#### Optuna TPE（ベースライン）

```bash
python scripts/optuna_tpe/run_experiment_NiB.py --seed 1
```

**特徴:**
- Tree-structured Parzen Estimator
- 100試行の最適化

#### 貪欲法

```bash
jupyter notebook notebooks/greedy_yield_NiB.ipynb
```

**特徴:**
- ReactionT5v2の予測収率上位から順に選択
- 探索なし、活用のみ

### 3. 結果の比較

```bash
jupyter notebook notebooks/compare_all_results.ipynb
```

全データセット・全手法の結果を集計し、以下を生成します:
- 最適化進捗の可視化（平均 ± 標準偏差）

## ノートブックの詳細

### 共通ノートブック

| ノートブック | 説明 |
|------------|------|
| `check_data.ipynb` | ReactionT5v2の学習データ（[ORD](https://drive.google.com/file/d/1JozA2OlByfZ-ILt5H5YrTjLJvSvD8xdL/view?usp=drive_link)）との類似性確認 |
| `compare_all_results.ipynb` | 全データセット・全手法の結果比較と可視化 |

### データセット別ノートブック

各データセット（NiB、SM、BH）に対して、以下の4つのノートブックが存在します:

| 種類 | 説明 | 特徴 |
|-----|------|------|
| `bo_yield_*.ipynb` | **ReactionT5 BO**（提案手法） | MC Dropout + UCB獲得関数 |
| `gpr_yield_*.ipynb` | **GPR BO**（ベースライン） | Morgan Fingerprint + GPR |
| `optuna_yield_*.ipynb` | **Optuna TPE**（ベースライン） | Tree-structured Parzen Estimator |
| `greedy_yield_*.ipynb` | **ReactionT5 貪欲法** | 予測収率上位から順に選択 |

**データセット:**
- `*_NiB.ipynb`: NiB（Nickel-catalyzed Borylation）データセット
- `*_SM.ipynb`: Suzuki-Miyauraデータセット
- `*_BH.ipynb`: Buchwald-Hartwigデータセット