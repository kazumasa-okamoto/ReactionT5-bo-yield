"""Data processing utilities for reaction yield optimization."""

from typing import Dict, List, Tuple

import pandas as pd
from rdkit import Chem

SMILES_SEPARATOR = "."
EMPTY_SMILES_TOKEN = " "


def space_clean(row: str) -> str:
    """Remove extra spaces around SMILES separators and trailing dots."""
    text = str(row)
    text = text.replace(". ", ".").replace(" .", ".").replace("  ", " ")
    # Remove consecutive dots
    while ".." in text:
        text = text.replace("..", ".")
    # Remove trailing dots
    text = text.rstrip(".")
    return text


def canonicalize(smiles: str) -> str:
    """Canonicalize a SMILES string."""
    try:
        molecule = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None
    except Exception:
        return None


def _normalize_multicomponent_smiles(smiles: str) -> str:
    """Sort dot-separated SMILES fragments into a deterministic order."""
    if not smiles or smiles == EMPTY_SMILES_TOKEN:
        return EMPTY_SMILES_TOKEN
    fragments = [fragment for fragment in smiles.split(SMILES_SEPARATOR) if fragment]
    return SMILES_SEPARATOR.join(sorted(fragments)) if fragments else EMPTY_SMILES_TOKEN


def _ensure_required_columns(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """Add any missing columns required for downstream processing."""
    for column in required_columns:
        if column not in df.columns:
            df[column] = EMPTY_SMILES_TOKEN
    return df


def _prepare_column(df: pd.DataFrame, column: str, canonicalize_entries: bool) -> pd.DataFrame:
    """Clean a column and optionally canonicalize entries."""
    df = df[~df[column].isna()].reset_index(drop=True)
    df[column] = df[column].fillna(EMPTY_SMILES_TOKEN).apply(space_clean)

    if canonicalize_entries:
        def _canonicalize_entry(value: str) -> str:
            if value == EMPTY_SMILES_TOKEN:
                return EMPTY_SMILES_TOKEN
            canonical = canonicalize(value)
            return canonical if canonical else EMPTY_SMILES_TOKEN

        df[column] = df[column].apply(_canonicalize_entry)
        df[column] = df[column].apply(_normalize_multicomponent_smiles)

    return df


def _combine_reagent_columns(df: pd.DataFrame) -> pd.Series:
    """Combine catalyst, reagent, and solvent columns into a single REAGENT column."""
    def merge_components(row):
        components = []
        for col in ["CATALYST", "REAGENT", "SOLVENT"]:
            val = row[col]
            if pd.notna(val) and str(val).strip() and str(val).strip() != EMPTY_SMILES_TOKEN:
                components.append(str(val).strip())
        return SMILES_SEPARATOR.join(components) if components else ""

    return df.apply(merge_components, axis=1)


def _canonicalize_components(smiles_str: str) -> str:
    """Canonicalize dot-separated SMILES components individually."""
    if not smiles_str or smiles_str.strip() == "" or smiles_str.strip() == EMPTY_SMILES_TOKEN:
        return ""

    components = smiles_str.split(SMILES_SEPARATOR)
    canonicalized = []

    for comp in components:
        comp = comp.strip()
        if comp and comp != EMPTY_SMILES_TOKEN:
            canon = canonicalize(comp)
            if canon:
                canonicalized.append(canon)

    return SMILES_SEPARATOR.join(sorted(canonicalized)) if canonicalized else ""


def load_and_preprocess_data(csv_path: str, dataset_name: str = "NiB") -> pd.DataFrame:
    """
    Load a dataset CSV and prepare it for Bayesian optimization.

    Args:
        csv_path: Path to the CSV file.
        dataset_name: Dataset identifier. Used for logging but preprocessing
            is now uniform across all datasets.

    Returns:
        A preprocessed DataFrame with canonicalized SMILES columns and
        normalized yields. The REAGENT column is created by combining
        CATALYST, REAGENT, and SOLVENT columns.

    Preprocessing logic:
        - NiB: REACTANT and PRODUCT are canonicalized. CATALYST, REAGENT,
          SOLVENT are NOT canonicalized but combined with dots.
        - SM/BH: REACTANT, PRODUCT, and REAGENT are canonicalized. CATALYST
          and SOLVENT are NOT canonicalized. Then CATALYST, REAGENT, SOLVENT
          are combined with dots.
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
        # Store the combined REAGENT in a temporary variable to avoid index mismatch
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


def create_reaction_dictionaries(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], Dict, Dict]:
    """
    Create lookup dictionaries for Bayesian optimization.

    Returns:
        Tuple containing sorted reactants, reagents, products, (reactant, reagent)
        to product mapping, and (reactant, reagent, product) to yield mapping.
    """
    reactant_list = sorted(df["REACTANT"].unique())
    reagent_list = sorted(df["REAGENT"].unique())
    product_list = sorted(df["PRODUCT"].unique())

    product_dict = {(row["REACTANT"], row["REAGENT"]): row["PRODUCT"] for _, row in df.iterrows()}
    true_yield_dict = {
        (row["REACTANT"], row["REAGENT"], row["PRODUCT"]): row["YIELD"] for _, row in df.iterrows()
    }

    print(f"Reactant candidates: {len(reactant_list)}")
    print(f"Reagent candidates: {len(reagent_list)}")
    print(f"Product candidates: {len(product_list)}")
    print(f"Known combinations: {len(product_dict)}")

    return reactant_list, reagent_list, product_list, product_dict, true_yield_dict
