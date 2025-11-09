"""Data loading utilities for Optuna TPE optimization experiments."""

import pandas as pd
from typing import Tuple, Dict, List


def load_buchwald_hartwig_data(data_path: str) -> Tuple[Dict, List, List, List, List]:
    """
    Load and prepare Buchwald-Hartwig dataset for optimization.

    Args:
        data_path: Path to CSV file

    Returns:
        Tuple of (yield_dict, ligand_list, additive_list, base_list, aryl_halide_list)
    """
    df = pd.read_csv(data_path)

    # Create lists
    ligand_list = df['Ligand'].unique().tolist()
    additive_list = df['Additive'].unique().tolist()
    base_list = df['Base'].unique().tolist()
    aryl_halide_list = df['Aryl halide'].unique().tolist()

    print(f"Dataset statistics:")
    print(f"  Ligand: {len(ligand_list)}")
    print(f"  Additive: {len(additive_list)}")
    print(f"  Base: {len(base_list)}")
    print(f"  Aryl halide: {len(aryl_halide_list)}")
    print(f"  Total combinations: {len(ligand_list) * len(additive_list) * len(base_list) * len(aryl_halide_list):,}")

    # Create yield dictionary
    keys = list(zip(df['Ligand'], df['Additive'], df['Base'], df['Aryl halide']))
    yield_dict = dict(zip(keys, df['Output']))

    return yield_dict, ligand_list, additive_list, base_list, aryl_halide_list


def load_nib_data(data_path: str) -> Tuple[Dict, List, List, List]:
    """
    Load and prepare NiB dataset for optimization.

    Args:
        data_path: Path to CSV file

    Returns:
        Tuple of (yield_dict, electrophile_list, ligand_list, solvent_list)
    """
    df = pd.read_csv(data_path)

    # Create lists
    electrophile_list = df['electrophile_inchi'].unique().tolist()
    ligand_list = df['ligand_inchi'].unique().tolist()
    solvent_list = df['solvent_inchi'].unique().tolist()

    print(f"Dataset statistics:")
    print(f"  Electrophiles: {len(electrophile_list)}")
    print(f"  Ligands: {len(ligand_list)}")
    print(f"  Solvents: {len(solvent_list)}")
    print(f"  Total combinations: {len(electrophile_list) * len(ligand_list) * len(solvent_list):,}")

    # Create yield dictionary
    keys = list(zip(df['electrophile_inchi'], df['ligand_inchi'], df['solvent_inchi']))
    yield_dict = dict(zip(keys, df['yield']))

    return yield_dict, electrophile_list, ligand_list, solvent_list


def load_suzuki_miyaura_data(data_path: str) -> Tuple[Dict, List, List, List, List, List, List]:
    """
    Load and prepare Suzuki-Miyaura dataset for optimization.

    Args:
        data_path: Path to CSV file

    Returns:
        Tuple of (yield_dict, reactant_1_list, reactant_2_list, catalyst_list,
                 ligand_list, reagent_list, solvent_list)
    """
    df = pd.read_csv(data_path)

    # Create lists
    reactant_1_name_list = df['Reactant_1_Name'].unique().tolist()
    reactant_2_name_list = df['Reactant_2_Name'].unique().tolist()
    catalyst_1_short_hand_list = df['Catalyst_1_Short_Hand'].unique().tolist()
    ligand_short_hand_list = df['Ligand_Short_Hand'].unique().tolist()
    reagent_1_short_hand_list = df['Reagent_1_Short_Hand'].unique().tolist()
    solvent_1_short_hand_list = df['Solvent_1_Short_Hand'].unique().tolist()

    print(f"Dataset statistics:")
    print(f"  Reactant_1_Name: {len(reactant_1_name_list)}")
    print(f"  Reactant_2_Name: {len(reactant_2_name_list)}")
    print(f"  Catalyst_1_Short_Hand: {len(catalyst_1_short_hand_list)}")
    print(f"  Ligand_Short_Hand: {len(ligand_short_hand_list)}")
    print(f"  Reagent_1_Short_Hand: {len(reagent_1_short_hand_list)}")
    print(f"  Solvent_1_Short_Hand: {len(solvent_1_short_hand_list)}")
    total = (len(reactant_1_name_list) * len(reactant_2_name_list) *
             len(catalyst_1_short_hand_list) * len(ligand_short_hand_list) *
             len(reagent_1_short_hand_list) * len(solvent_1_short_hand_list))
    print(f"  Total combinations: {total:,}")

    # Create yield dictionary
    keys = list(zip(df['Reactant_1_Name'], df['Reactant_2_Name'],
                    df['Catalyst_1_Short_Hand'], df['Ligand_Short_Hand'],
                    df['Reagent_1_Short_Hand'], df['Solvent_1_Short_Hand']))
    yield_dict = dict(zip(keys, df['Product_Yield_PCT_Area_UV']))

    return (yield_dict, reactant_1_name_list, reactant_2_name_list,
            catalyst_1_short_hand_list, ligand_short_hand_list,
            reagent_1_short_hand_list, solvent_1_short_hand_list)
