[ochem-data](https://github.com/doyle-lab-ucla/ochem-data)の[NiB](https://github.com/doyle-lab-ucla/ochem-data/tree/main/NiB)データセットと、[rxn_yields](https://github.com/rxn4chemistry/rxn_yields)の[Suzuki-Miyaura](https://github.com/rxn4chemistry/rxn_yields/tree/master/data/Suzuki-Miyaura)データセット、[Buchwald-Hartwig](https://github.com/rxn4chemistry/rxn_yields/tree/master/data/Buchwald-Hartwig)データセットを実際の実験結果として、[ReactionT5v2](https://github.com/sagawatatsuya/ReactionT5v2?tab=readme-ov-file)を用いてその中から収率の大きい反応を探索します

## ノートブックの概要
**NiBデータセット**
- greedy_optuna_NiB.ipynb：NiBデータセットに対して、ReactionT5v2で高い収率が予測される反応をOptunaを用いて探索します

- bo_yield_NiB.ipynb：NiBデータセットに対して、ReactionT5v2にMC Dropoutを適用することで収率予測の平均と分散を出力し、これを用いてベイズ最適化により収率の高い反応を探索します

- greedy_yield.ipynb_NiB：NiBデータセットに対して、貪欲法を用いてReactionT5v2が高い収率を予測した反応から探索します

**Suzuki-Miyauraデータセット**
- bo_yield_SM.ipynb：Suzuki-Miyauraデータセットに対して、ReactionT5v2にMC Dropoutを適用することで収率予測の平均と分散を出力し、これを用いてベイズ最適化により収率の高い反応を探索します

- greedy_yield_SM：Suzuki-Miyauraデータセットに対して、貪欲法を用いてReactionT5v2が高い収率を予測した反応から探索します