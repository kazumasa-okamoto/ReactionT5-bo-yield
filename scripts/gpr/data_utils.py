"""Data processing utilities for GPR yield optimization."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

EMPTY_TOKEN = ""
SMILES_SEPARATOR = "."


def space_clean(value) -> str:
    """Remove extra spaces around SMILES separators and trailing dots."""
    text = str(value)
    text = text.replace(". ", SMILES_SEPARATOR).replace(" .", SMILES_SEPARATOR).replace("  ", " ")
    # Remove consecutive dots
    while ".." in text:
        text = text.replace("..", ".")
    # Remove trailing dots
    text = text.rstrip(".")
    return text


def canonicalize(smiles: str) -> str:
    """Canonicalize a SMILES string; return None if parsing fails."""
    try:
        molecule = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None
    except Exception:
        return None


def _normalize_multicomponent_smiles(smiles: str) -> str:
    """Sort dot-separated SMILES fragments into a deterministic order."""
    if not smiles or smiles == EMPTY_TOKEN:
        return EMPTY_TOKEN
    fragments = [frag for frag in smiles.split(SMILES_SEPARATOR) if frag]
    return SMILES_SEPARATOR.join(sorted(fragments)) if fragments else EMPTY_TOKEN


def _ensure_required_columns(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """Ensure that all required columns exist; missing ones are filled with EMPTY_TOKEN."""
    for column in required_columns:
        if column not in df.columns:
            df[column] = EMPTY_TOKEN
    return df


def _prepare_column(df: pd.DataFrame, column: str, canonicalize_entries: bool) -> pd.DataFrame:
    """Clean a column and optionally canonicalize entries."""
    df[column] = df[column].fillna(EMPTY_TOKEN).apply(space_clean)

    if canonicalize_entries:
        def _canonicalize_entry(value: str) -> str:
            if value == EMPTY_TOKEN:
                return EMPTY_TOKEN
            canonical = canonicalize(value)
            return canonical if canonical else EMPTY_TOKEN

        df[column] = df[column].apply(_canonicalize_entry)
        df[column] = df[column].apply(_normalize_multicomponent_smiles)

    return df


def _combine_reagent_columns(df: pd.DataFrame) -> pd.Series:
    """Combine catalyst, reagent, and solvent columns into a single REAGENT column."""
    def merge_components(row):
        components = []
        for col in ["CATALYST", "REAGENT", "SOLVENT"]:
            val = row[col]
            if pd.notna(val) and str(val).strip() and str(val).strip() != EMPTY_TOKEN:
                components.append(str(val).strip())
        return SMILES_SEPARATOR.join(components) if components else ""

    return df.apply(merge_components, axis=1)


def _canonicalize_components(smiles_str: str) -> str:
    """Canonicalize dot-separated SMILES components individually."""
    if not smiles_str or smiles_str.strip() == "" or smiles_str.strip() == EMPTY_TOKEN:
        return ""

    components = smiles_str.split(SMILES_SEPARATOR)
    canonicalized = []

    for comp in components:
        comp = comp.strip()
        if comp and comp != EMPTY_TOKEN:
            canon = canonicalize(comp)
            if canon:
                canonicalized.append(canon)

    return SMILES_SEPARATOR.join(sorted(canonicalized)) if canonicalized else ""


def load_and_preprocess_data(csv_path: str, dataset_name: str = "NiB") -> pd.DataFrame:
    """
    Load and preprocess reaction data for GPR optimization.

    Args:
        csv_path: Path to the dataset CSV.
        dataset_name: Dataset identifier (e.g., "NiB", "SM", "BH").

    Returns:
        Preprocessed DataFrame with canonicalized SMILES strings and normalized yields.
    """
    df = pd.read_csv(csv_path)
    df = _ensure_required_columns(df, ["REACTANT", "CATALYST", "REAGENT", "SOLVENT", "PRODUCT"])

    if "YIELD" in df.columns:
        df["YIELD"] = df["YIELD"].clip(0, 100) / 100

    if dataset_name == "NiB":
        # NiB: Canonicalize REACTANT and PRODUCT only
        for column in ["REACTANT", "PRODUCT"]:
            df = _prepare_column(df, column, canonicalize_entries=True)
        # Combine CATALYST, REAGENT, SOLVENT without canonicalization
        combined_reagent = _combine_reagent_columns(df)
        df = df.loc[combined_reagent.index].reset_index(drop=True)
        df["REAGENT"] = combined_reagent.reset_index(drop=True)

        # Clean and canonicalize REAGENT components
        df["REAGENT"] = df["REAGENT"].apply(space_clean)
        df["REAGENT"] = df["REAGENT"].apply(_canonicalize_components)

        # Remove rows with empty REAGENT
        df = df[df["REAGENT"] != ""].reset_index(drop=True)
    else:
        # SM/BH: Canonicalize REACTANT, PRODUCT, and REAGENT
        for column in ["REAGENT", "REACTANT", "PRODUCT"]:
            df = _prepare_column(df, column, canonicalize_entries=True)

        # Combine CATALYST, REAGENT, SOLVENT
        combined_reagent = _combine_reagent_columns(df)
        df = df.loc[combined_reagent.index].reset_index(drop=True)
        df["REAGENT"] = combined_reagent.reset_index(drop=True)

        # Clean and canonicalize REAGENT components
        df["REAGENT"] = df["REAGENT"].apply(space_clean)
        df["REAGENT"] = df["REAGENT"].apply(_canonicalize_components)

        # Remove rows with empty REAGENT
        df = df[df["REAGENT"] != ""].reset_index(drop=True)

    return df


def smiles_to_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Convert SMILES to Morgan Fingerprint using rdFingerprintGenerator.

    Args:
        smiles: SMILES string
        radius: Morgan Fingerprint radius (default: 2)
        n_bits: Number of bits (default: 2048)

    Returns:
        numpy array: Fingerprint bit vector
    """
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return np.zeros(n_bits, dtype=np.float32)
        # Generate Morgan fingerprint using rdFingerprintGenerator
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fingerprint = fpgen.GetFingerprint(molecule)
        # Convert to numpy array
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, arr)
        return arr
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


def reaction_to_fingerprint(reactant: str, reagent: str, product: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Combine Morgan Fingerprints from reactant, reagent, and product.

    Args:
        reactant: Reactant SMILES
        reagent: Reagent SMILES
        product: Product SMILES
        radius: Morgan Fingerprint radius
        n_bits: Number of bits per molecule fingerprint

    Returns:
        numpy array: Concatenated fingerprint
    """
    fp_reactant = smiles_to_morgan_fingerprint(reactant, radius, n_bits)
    fp_reagent = smiles_to_morgan_fingerprint(reagent, radius, n_bits)
    fp_product = smiles_to_morgan_fingerprint(product, radius, n_bits)

    # Concatenate three fingerprints
    return np.concatenate([fp_reactant, fp_reagent, fp_product])


def compute_all_fingerprints(df: pd.DataFrame, radius: int = 2, n_bits: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Morgan Fingerprints for all reactions in dataframe.

    Args:
        df: DataFrame with REACTANT, REAGENT, PRODUCT columns
        radius: Morgan Fingerprint radius
        n_bits: Number of bits per molecule

    Returns:
        tuple: (fingerprints array, yields array)
    """
    print("Computing Morgan fingerprints...")
    fingerprints = []
    for idx, row in df.iterrows():
        fp = reaction_to_fingerprint(row["REACTANT"], row["REAGENT"], row["PRODUCT"], radius, n_bits)
        fingerprints.append(fp)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} of {len(df)} reactions")

    X = np.array(fingerprints)
    y = df["YIELD"].values * 100  # Convert to percentage

    print(f"\nFingerprint matrix shape: {X.shape}")
    print(f"Yield range: {y.min():.2f}% - {y.max():.2f}%")

    return X, y


def create_reaction_dictionaries(df: pd.DataFrame, X: np.ndarray) -> Tuple[List[str], List[str], List[str], Dict, Dict, Dict]:
    """
    Create reaction dictionaries from dataset.

    Args:
        df: DataFrame
        X: Fingerprints array

    Returns:
        tuple: (reactant_list, reagent_list, product_list, product_dict, true_yield_dict, fingerprint_dict)
    """
    reactant_list = sorted(df["REACTANT"].unique())
    reagent_list = sorted(df["REAGENT"].unique())
    product_list = sorted(df["PRODUCT"].unique())

    # (reactant, reagent) -> product mapping
    product_dict = {
        (row["REACTANT"], row["REAGENT"]): row["PRODUCT"]
        for _, row in df.iterrows()
    }

    # (reactant, reagent, product) -> yield mapping
    true_yield_dict = {
        (row["REACTANT"], row["REAGENT"], row["PRODUCT"]): row["YIELD"] * 100
        for _, row in df.iterrows()
    }

    # (reactant, reagent, product) -> fingerprint mapping
    fingerprint_dict = {}
    for idx, row in df.iterrows():
        key = (row["REACTANT"], row["REAGENT"], row["PRODUCT"])
        fingerprint_dict[key] = X[idx]

    print(f"Reactant candidates: {len(reactant_list)}")
    print(f"Reagent candidates: {len(reagent_list)}")
    print(f"Product candidates: {len(product_list)}")
    print(f"Known combinations: {len(product_dict)}")

    return reactant_list, reagent_list, product_list, product_dict, true_yield_dict, fingerprint_dict
