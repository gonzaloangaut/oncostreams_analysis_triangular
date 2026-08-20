from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Read the information encoded in the file names
KNOWN_FIELDS = [
    "initial_number_of_cells",
    "initial_fraction_elongated",
    "requested_density",
    "target_density",
    "density",
    "requested_nc",
    "reference_nc",
    "initial_nc",
    "target_nc",
    "removed_nc",
    "initial_f_e",
    "force",
    "rng_seed",
]

FIELD_PATTERN = re.compile(
    r"_(?P<key>" + "|".join(sorted(KNOWN_FIELDS, key=len, reverse=True)) + r")="
    r"(?P<value>.*?)"
    r"(?=_(?:" + "|".join(sorted(KNOWN_FIELDS, key=len, reverse=True)) + r")=|$)"
)

STEP_PATTERN = re.compile(r"_step=(\d+)\.dat$")


# Order parameter columns that should be present in every file
ORDER_PARAMETER_COLUMNS = [
    "nematic",
    "polar",
    "nematic_2",
    "polar_2",
    "fraction_elongated",
]

# Create a function to extract the information
def parse_metadata(filename: str) -> dict:
    """Extract simulation metadata encoded in an output filename."""

    # Get the step
    step_match = STEP_PATTERN.search(filename)

    if step_match is None:
        raise ValueError(f"Could not read step from filename: {filename}")

    step = int(step_match.group(1))
    name_without_step = STEP_PATTERN.sub("", filename)

    # Search every field in the filename
    fields = {
        match.group("key"): match.group("value")
        for match in FIELD_PATTERN.finditer(name_without_step)
    }

    # Seed
    seed = int(fields["rng_seed"])

    # Number of cells
    requested_n = fields.get(
        "requested_nc",
        fields.get("target_nc"),
    )

    initial_n = fields.get(
        "initial_nc",
        fields.get("initial_number_of_cells"),
    )

    reference_n = fields.get("reference_nc")

    if requested_n is None:
        requested_n = initial_n

    if initial_n is None:
        initial_n = requested_n

    if reference_n is None:
        reference_n = requested_n

    # Density
    requested_rho = fields.get(
        "requested_density",
        fields.get("target_density"),
    )

    actual_rho = fields.get("density")

    if requested_rho is None:
        requested_rho = actual_rho

    if actual_rho is None:
        actual_rho = requested_rho

    # Initial elongated fraction
    initial_fraction_elongated = fields.get(
        "initial_f_e",
        fields.get("initial_fraction_elongated", "0"),
    )

    if requested_n is None or requested_rho is None:
        raise ValueError(
            "Could not identify N and density from filename: "
            f"{filename}"
        )

    return {
        "N": int(requested_n),
        "reference_N": int(reference_n),
        "actual_N": int(initial_n),
        "rho": float(requested_rho),
        "actual_rho": float(actual_rho),
        "seed": seed,
        "step": step,
        "initial_fraction_elongated": float(initial_fraction_elongated),
        "force": fields.get("force", ""),
    }

# Process one order-parameter file
def process_order_parameter_file(filepath: Path) -> dict:
    """
    Read one order-parameter snapshot and return one row containing
    simulation metadata and global order parameters.
    """

    # Get the metadata from the filename
    metadata = parse_metadata(filepath.name)

    data = pd.read_csv(
        filepath,
        skipinitialspace=True,
    )

    # Every file should contain exactly one snapshot
    if len(data) != 1:
        raise ValueError(
            f"Expected exactly one row in {filepath}, "
            f"found {len(data)}."
        )

    # Check expected columns
    missing_columns = [
        column
        for column in ORDER_PARAMETER_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {filepath}: {missing_columns}"
        )

    observables = (
        data.iloc[0][ORDER_PARAMETER_COLUMNS]
        .to_dict()
    )

    return {
        **metadata,
        **observables,
    }


# Process all files
def process_order_parameter_files(
    op_files: list[Path],
) -> pd.DataFrame:
    """Combine all order-parameter snapshots into one DataFrame."""

    rows = []

    for index, filepath in enumerate(op_files, start=1):

        rows.append(
            process_order_parameter_file(filepath)
        )

        if index % 1000 == 0 or index == len(op_files):
            print(
                f"Order parameter files processed: "
                f"{index}/{len(op_files)}"
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# Sanity checks
def validate_order_parameters(data: pd.DataFrame) -> None:
    """
    Check the identities

        polar_2   = fraction_elongated * polar
        nematic_2 = fraction_elongated * nematic

    which should hold up to numerical precision.
    """

    expected_polar_2 = (
        data["fraction_elongated"]
        * data["polar"]
    )

    expected_nematic_2 = (
        data["fraction_elongated"]
        * data["nematic"]
    )

    polar_ok = np.allclose(
        data["polar_2"],
        expected_polar_2,
        equal_nan=True,
    )

    nematic_ok = np.allclose(
        data["nematic_2"],
        expected_nematic_2,
        equal_nan=True,
    )

    polar_error = np.nanmax(
        np.abs(
            data["polar_2"]
            - expected_polar_2
        )
    )

    nematic_error = np.nanmax(
        np.abs(
            data["nematic_2"]
            - expected_nematic_2
        )
    )

    print("\nSanity checks:")
    print(
        "polar_2 = fraction_elongated * polar:",
        polar_ok,
    )
    print(
        "nematic_2 = fraction_elongated * nematic:",
        nematic_ok,
    )

    print(
        "Maximum polar identity error:",
        polar_error,
    )
    print(
        "Maximum nematic identity error:",
        nematic_error,
    )

# Main
def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Consolidate global order-parameter output files "
            "into one parquet dataset."
        )
    )

    parser.add_argument(
        "data_root",
        type=Path,
        help="Root directory containing simulation outputs.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<data_root>/processed/order_parameters"
        ),
    )

    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_root / "processed" / "order_parameters"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Find all global order-parameter files
    op_files = sorted(
        data_root.rglob("op_culture_*.dat")
    )

    print(
        f"Order parameter files found: {len(op_files)}"
    )

    if not op_files:
        raise FileNotFoundError(
            f"No order parameter files found below {data_root}"
        )

    # Process data
    order_parameters = process_order_parameter_files(
        op_files
    )

    # Sort data
    sort_columns = [
        "N",
        "rho",
        "seed",
        "step",
    ]

    order_parameters = (
        order_parameters
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )

    # Validate relationships between observables
    validate_order_parameters(
        order_parameters
    )

    # Save
    output_path = (
        output_dir
        / "order_parameters.parquet"
    )

    order_parameters.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"\nProcessed order parameter data written to:"
        f"\n{output_path}"
    )

    print(
        f"\nRows written: {len(order_parameters)}"
    )


if __name__ == "__main__":
    main()