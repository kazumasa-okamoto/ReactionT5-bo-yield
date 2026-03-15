"""Data processing utilities for GNN-based reaction yield optimization."""

import os
import shutil
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from tqdm import tqdm
import glob


def space_clean(row):
    """Clean spaces in SMILES string."""
    row = row.replace(". ", "").replace(" .", "").replace("  ", " ")
    return row


def canonicalize(smiles):
    """Canonicalize SMILES string."""
    try:
        new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    except:
        new_smiles = None
    return new_smiles


def get_global_symbols(df, smiles_cols):
    """Get all atomic symbols present in the dataset."""
    symbols = set()
    for col in smiles_cols:
        if col in df.columns:
            for sm in df[col].dropna():
                mol = Chem.MolFromSmiles(sm)
                if mol:
                    for atom in mol.GetAtoms():
                        symbols.add(atom.GetSymbol())
    return sorted(list(symbols))


def get_atom_features(atom, symbol_list):
    """Get atom feature vector."""
    # One-hot encoding for atom symbol
    symbol = [1 if atom.GetSymbol() == s else 0 for s in symbol_list]

    features = symbol + [
        atom.GetDegree(),
        atom.GetTotalValence(),
        atom.GetIsAromatic() * 1.0,
    ]
    return torch.tensor(features, dtype=torch.float)


def get_bond_features(bond):
    """Get bond feature vector."""
    bt = bond.GetBondType()
    features = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
    ]
    return torch.tensor(features, dtype=torch.float)


def smiles_to_pyg_graph(smiles_list, symbol_list, yield_val=None):
    """Convert SMILES list to PyTorch Geometric graph."""
    all_atom_feats = []
    all_edge_indices = []
    all_edge_feats = []
    current_node_offset = 0

    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)

        # 1. Get atom features
        for atom in mol.GetAtoms():
            all_atom_feats.append(get_atom_features(atom, symbol_list))

        # 2. Get bond features
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            # Undirected graph - add both directions
            all_edge_indices.append([start + current_node_offset, end + current_node_offset])
            all_edge_indices.append([end + current_node_offset, start + current_node_offset])

            b_feat = get_bond_features(bond)
            all_edge_feats.append(b_feat)
            all_edge_feats.append(b_feat)

        current_node_offset += mol.GetNumAtoms()

    # Guard against empty graphs
    if not all_atom_feats:
        return None

    return Data(
        x=torch.stack(all_atom_feats),
        edge_index=torch.tensor(all_edge_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(all_edge_feats),
        y=torch.tensor([yield_val], dtype=torch.float) if yield_val is not None else None
    )


def load_and_preprocess_data(csv_path: str, dataset_name: str = "BH"):
    """Load and preprocess reaction data."""
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_cols = ["REACTANT", "CATALYST", "REAGENT", "SOLVENT", "PRODUCT"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    # Normalize YIELD to 0-1 range
    if "YIELD" in df.columns:
        df["YIELD"] = df["YIELD"].clip(0, 100) / 100

    # Combine CATALYST, REAGENT, SOLVENT into single REAGENT column
    df["REAGENT"] = (
        df["CATALYST"].fillna(" ") + "." +
        df["REAGENT"].fillna(" ") + "." +
        df["SOLVENT"].fillna(" ")
    )

    # Clean and canonicalize
    for col in ["REAGENT", "REACTANT", "PRODUCT"]:
        df[col] = df[col].apply(space_clean)
        df[col] = df[col].apply(lambda x: canonicalize(x) if x != " " else " ")
        df = df[~df[col].isna()].reset_index(drop=True)
        df[col] = df[col].apply(lambda x: ".".join(sorted(x.split("."))))

    return df


def process_and_save_graphs(df, save_root):
    """Process all reactions and save as PyG graphs."""
    smiles_cols = ["REACTANT", "REAGENT", "PRODUCT"]
    target_col = "YIELD"

    # 1. Reset save directory
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.makedirs(save_root)

    # 2. Get global atomic symbols
    global_symbol_list = get_global_symbols(df, smiles_cols)
    print(f"Global Atomic Symbols ({len(global_symbol_list)}): {global_symbol_list}")

    # 3. Convert to graphs and save
    idx = 0
    index_map = []

    print(f"Processing dataset...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sm_list = [row[c] for c in smiles_cols if c in df.columns and pd.notna(row[c])]
        yield_val = row[target_col] if target_col in df.columns else 0.0

        try:
            graph = smiles_to_pyg_graph(sm_list, global_symbol_list, yield_val)
            if graph is not None:
                save_path = os.path.join(save_root, f'data_{idx}.pt')
                torch.save(graph, save_path)
                index_map.append({"idx": idx})
                idx += 1
        except Exception:
            # Skip invalid SMILES
            continue

    # Save metadata
    torch.save(global_symbol_list, os.path.join(save_root, 'symbols.pt'))
    print(f"Finished. Total graphs saved: {idx}")
    return global_symbol_list, idx


class ReactionDataset(Dataset):
    """Dataset for reaction graphs."""

    def __init__(self, root, transform=None, pre_transform=None):
        """
        Args:
            root: Directory containing .pt files
        """
        super().__init__(root, transform, pre_transform)

        # List all data_*.pt files
        self.file_list = glob.glob(os.path.join(self.root, 'data_*.pt'))
        self.total_count = len(self.file_list)

        if self.total_count == 0:
            print(f"Warning: No data files found in {self.root}")
        else:
            print(f"Successfully loaded dataset with {self.total_count} graphs.")

    def len(self):
        return self.total_count

    def get(self, idx):
        path = os.path.join(self.root, f'data_{idx}.pt')
        data = torch.load(path, weights_only=False)
        return data
