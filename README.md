[ochem-data](https://github.com/doyle-lab-ucla/ochem-data)の[NiB](https://github.com/doyle-lab-ucla/ochem-data/tree/main/NiB)データセットを実際の実験結果として、[ReactionT5v2](https://github.com/sagawatatsuya/ReactionT5v2?tab=readme-ov-file)を用いてその中から収率の大きい反応を探索します

- greedy_optuna.ipynb：ReactionT5v2で高い収率が予測される反応をOptunaを用いて探索します

- bo_yield.ipynb：ReactionT5v2にMC Dropoutを適用することで収率予測の平均と分散を出力し、これを用いてベイズ最適化により収率の高い反応を探索します