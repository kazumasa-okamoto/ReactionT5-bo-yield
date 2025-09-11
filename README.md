[ochem-data](https://github.com/doyle-lab-ucla/ochem-data)の[NiB](https://github.com/doyle-lab-ucla/ochem-data/tree/main/NiB)データセットと、[rxn_yields](https://github.com/rxn4chemistry/rxn_yields)の[Suzuki-Miyaura](https://github.com/rxn4chemistry/rxn_yields/tree/master/data/Suzuki-Miyaura)データセットを実際の実験結果として、[ReactionT5v2](https://github.com/sagawatatsuya/ReactionT5v2?tab=readme-ov-file)を用いてその中から収率の大きい反応を探索します

- greedy_optuna.ipynb：NiBデータセットに対して、ReactionT5v2で高い収率が予測される反応をOptunaを用いて探索します

- bo_yield.ipynb：NiBデータセットに対して、ReactionT5v2にMC Dropoutを適用することで収率予測の平均と分散を出力し、これを用いてベイズ最適化により収率の高い反応を探索します

- greedy_yield.ipynb：NiBデータセットに対して、貪欲法を用いてReactionT5v2が高い収率を予測した反応から探索します

- bo_yield_aap9112.ipynb：Suzuki-Miyauraデータセットに対して、ReactionT5v2にMC Dropoutを適用することで収率予測の平均と分散を出力し、これを用いてベイズ最適化により収率の高い反応を探索します

- greedy_yield_aap9112：Suzuki-Miyauraデータセットに対して、貪欲法を用いてReactionT5v2が高い収率を予測した反応から探索します