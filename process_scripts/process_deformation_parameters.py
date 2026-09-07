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


# Columns that should be present in every deformation file
DEFORMATION_COUNT_COLUMNS = [
    "round_elongation_attempts",
    "round_elongation_successes",
    "elliptical_elongation_attempts",
    "elliptical_elongation_successes",
    "contraction_events",
    "contraction_to_round_events",
    "contraction_overlap_rejections",
]


DEFORMATION_MAX_COLUMNS = [
    "max_contraction_proposed_overlap",
    "max_contraction_accepted_overlap",
]


DEFORMATION_COLUMNS = [
    "tic_start",
    "tic_end",
    "number_of_steps",
    *DEFORMATION_COUNT_COLUMNS,
    *DEFORMATION_MAX_COLUMNS,
]


def parse_metadata(filename: str) -> dict:
    """Extract simulation metadata encoded in an output filename."""

    step_match = STEP_PATTERN.search(filename)

    if step_match is None:
        raise ValueError(f"Could not read step from filename: {filename}")

    step = int(step_match.group(1))
    name_without_step = STEP_PATTERN.sub("", filename)

    fields = {
        match.group("key"): match.group("value")
        for match in FIELD_PATTERN.finditer(name_without_step)
    }

    seed = int(fields["rng_seed"])

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

    requested_rho = fields.get(
        "requested_density",
        fields.get("target_density"),
    )

    actual_rho = fields.get("density")

    if requested_rho is None:
        requested_rho = actual_rho

    if actual_rho is None:
        actual_rho = requested_rho

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


def process_deformation_file(filepath: Path) -> dict:
    """
    Read one deformation-parameter file and return one processed row.

    The number of contraction attempts and the rejection fraction are
    calculated from accepted contractions and safety-check rejections.
    """

    metadata = parse_metadata(filepath.name)

    data = pd.read_csv(
        filepath,
        skipinitialspace=True,
    )

    if len(data) != 1:
        raise ValueError(
            f"Expected exactly one row in {filepath}, "
            f"found {len(data)}."
        )

    missing_columns = [
        column
        for column in DEFORMATION_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {filepath}: {missing_columns}"
        )

    row = data.iloc[0]

    interval_data = {
        "tic_start": int(row["tic_start"]),
        "tic_end": int(row["tic_end"]),
        "number_of_steps": int(row["number_of_steps"]),
    }

    count_data = {
        column: int(row[column])
        for column in DEFORMATION_COUNT_COLUMNS
    }

    maximum_data = {
        column: float(row[column])
        for column in DEFORMATION_MAX_COLUMNS
    }

    contraction_attempts = (
        count_data["contraction_events"]
        + count_data["contraction_overlap_rejections"]
    )

    contraction_rejection_fraction = (
        count_data["contraction_overlap_rejections"]
        / contraction_attempts
        if contraction_attempts > 0
        else np.nan
    )

    return {
        **metadata,
        **interval_data,
        **count_data,
        **maximum_data,
        "contraction_attempts": contraction_attempts,
        "contraction_rejection_fraction": (
            contraction_rejection_fraction
        ),
    }


def process_deformation_files(
    deformation_files: list[Path],
) -> pd.DataFrame:
    """Combine all deformation intervals into one DataFrame."""

    rows = []

    for index, filepath in enumerate(deformation_files, start=1):
        rows.append(
            process_deformation_file(filepath)
        )

        if index % 1000 == 0 or index == len(deformation_files):
            print(
                f"Deformation files processed: "
                f"{index}/{len(deformation_files)}"
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def validate_deformation(data: pd.DataFrame) -> None:
    """Perform consistency checks on the deformation intervals."""

    duplicated = data.duplicated(
        subset=["N", "rho", "seed", "step"],
    ).sum()

    step_mismatches = int(
        (data["step"] != data["tic_end"]).sum()
    )

    expected_number_of_steps = (
        data["tic_end"]
        - data["tic_start"]
        + 1
    )

    interval_length_mismatches = int(
        (
            data["number_of_steps"]
            != expected_number_of_steps
        ).sum()
    )

    negative_counts = {
        column: int((data[column] < 0).sum())
        for column in (
            DEFORMATION_COUNT_COLUMNS
            + ["contraction_attempts"]
        )
    }

    invalid_rejection_fractions = int(
        (
            data["contraction_rejection_fraction"].notna()
            & ~data["contraction_rejection_fraction"].between(0, 1)
        ).sum()
    )

    negative_maxima = {
        column: int((data[column] < 0).sum())
        for column in DEFORMATION_MAX_COLUMNS
    }

    nonfinite_maxima = {
        column: int((~np.isfinite(data[column])).sum())
        for column in DEFORMATION_MAX_COLUMNS
    }

    accepted_larger_than_proposed = int(
        (
            data["max_contraction_accepted_overlap"]
            > data["max_contraction_proposed_overlap"] + 1e-12
        ).sum()
    )

    print("\nSanity checks:")
    print(
        "Duplicated (N, rho, seed, step) rows:",
        duplicated,
    )
    print(
        "Rows where filename step differs from tic_end:",
        step_mismatches,
    )
    print(
        "Rows with an inconsistent interval length:",
        interval_length_mismatches,
    )

    for column, number_negative in negative_counts.items():
        print(
            f"Negative values in {column}:",
            number_negative,
        )

    for column, number_negative in negative_maxima.items():
        print(
            f"Negative values in {column}:",
            number_negative,
        )

    for column, number_nonfinite in nonfinite_maxima.items():
        print(
            f"Non-finite values in {column}:",
            number_nonfinite,
        )

    print(
        "Invalid contraction rejection fractions:",
        invalid_rejection_fractions,
    )
    print(
        "Rows where accepted maximum exceeds proposed maximum:",
        accepted_larger_than_proposed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate deformation-parameter output files "
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
            "<data_root>/processed/deformation"
        ),
    )

    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_root / "processed" / "deformation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    deformation_files = sorted(
        data_root.rglob("deformation_events_*.dat")
    )

    print(
        f"Deformation files found: {len(deformation_files)}"
    )

    if not deformation_files:
        raise FileNotFoundError(
            f"No deformation files found below {data_root}"
        )

    deformation = process_deformation_files(
        deformation_files
    )

    sort_columns = [
        "N",
        "rho",
        "seed",
        "step",
    ]

    deformation = (
        deformation
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )

    validate_deformation(
        deformation
    )

    output_path = (
        output_dir
        / "deformation.parquet"
    )

    deformation.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"\nProcessed deformation data written to:"
        f"\n{output_path}"
    )

    print(
        f"\nRows written: {len(deformation)}"
    )


if __name__ == "__main__":
    main()
