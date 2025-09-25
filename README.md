# 事前学習済みTransformer（ReactionT5v2）を用いた化学反応収率のベイズ最適化
[ochem-data](https://github.com/doyle-lab-ucla/ochem-data)の[NiB](https://github.com/doyle-lab-ucla/ochem-data/tree/main/NiB)データセットと、[rxn_yields](https://github.com/rxn4chemistry/rxn_yields)の[Suzuki-Miyaura](https://github.com/rxn4chemistry/rxn_yields/tree/master/data/Suzuki-Miyaura)データセット、[Buchwald-Hartwig](https://github.com/rxn4chemistry/rxn_yields/tree/master/data/Buchwald-Hartwig)データセットを実際の実験結果として、[ReactionT5v2](https://github.com/sagawatatsuya/ReactionT5v2?tab=readme-ov-file)を用いたベイズ最適化によってその中から収率の高い反応条件を探索します

## ノートブックの概要
**NiBデータセット**

- bo_yield_NiB.ipynb：NiBデータセットに対して、ReactionT5v2にMC Dropoutを適用することで収率予測の平均と分散を出力し、これを用いてベイズ最適化により収率の高い反応条件を探索します

- greedy_yield_NiB.ipynb：NiBデータセットに対して、貪欲法を用いてReactionT5v2が高い収率を予測した反応から探索します

- optuna_yeild_NiB.ipynb：NiBデータセットに対して、Optunaを用いて収率の高い反応条件を探索します。TPESamplerを用いた場合、GPSamplerを用いた場合をそれぞれ実験しています。この結果をベースラインとして使用します。

**Suzuki-Miyauraデータセット**
- bo_yield_SM.ipynb：Suzuki-Miyauraデータセットに対して、ReactionT5v2にMC Dropoutを適用することで収率予測の平均と分散を出力し、これを用いてベイズ最適化により収率の高い反応条件を探索します

- greedy_yield_SM.ipynb：Suzuki-Miyauraデータセットに対して、貪欲法を用いてReactionT5v2が高い収率を予測した反応から探索します

- optuna_yeild_SM.ipynb：Suzuki-Miyauraデータセットに対して、Optunaを用いて収率の高い反応条件を探索します。TPESamplerを用いた場合、GPSamplerを用いた場合をそれぞれ実験しています。この結果をベースラインとして使用します。

**Buchwald-Hartwigデータセット**
- bo_yield_BH.ipynb：Buchwald-Hartwigデータセットに対して、ReactionT5v2にMC Dropoutを適用することで収率予測の平均と分散を出力し、これを用いてベイズ最適化により収率の高い反応条件を探索します

- greedy_yield_BH.ipynb：Buchwald-Hartwigデータセットに対して、貪欲法を用いてReactionT5v2が高い収率を予測した反応から探索します

- optuna_yeild_BH.ipynb：Buchwald-Hartwigデータセットに対して、Optunaを用いて収率の高い反応条件を探索します。TPESamplerを用いた場合、GPSamplerを用いた場合をそれぞれ実験しています。この結果をベースラインとして使用します。