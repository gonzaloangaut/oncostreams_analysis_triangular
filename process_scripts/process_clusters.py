from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Read the information of the name of the files
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

# Create a function to extract the information
def parse_metadata(filename: str) -> dict:
    """Extract the simulation metadata encoded in an output filename."""
    # Get the step
    step_match = STEP_PATTERN.search(filename)
    if step_match is None:
        raise ValueError(f"Could not read step from filename: {filename}")

    step = int(step_match.group(1))
    name_without_step = STEP_PATTERN.sub("", filename)

    # Search every fields of the name
    fields = {
        match.group("key"): match.group("value")
        for match in FIELD_PATTERN.finditer(name_without_step)
    }

    # And get the data for every parameter
    seed = int(fields["rng_seed"])

    requested_n = fields.get("requested_nc", fields.get("target_nc"))
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

# Create a function to add data to a table
def add_metadata(data: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Add simulation metadata columns to one output table."""
    data = data.copy()
    for key, value in metadata.items():
        data[key] = value

    metadata_columns = [
        "N",
        "reference_N",
        "actual_N",
        "rho",
        "actual_rho",
        "seed",
        "step",
        "initial_fraction_elongated",
        "force",
    ]

    other_columns = [
        column
        for column in data.columns
        if column not in metadata_columns
    ]

    return data[metadata_columns + other_columns]

# Create a function to analize each snapshot
def calculate_snapshot_statistics(
    raw_data: pd.DataFrame,
    metadata: dict,
    phenotype: str,
) -> dict:
    """Calculate cluster observables for one phenotype and one snapshot."""
    # Take one phenotype
    phenotype_data = raw_data.loc[
        raw_data["phenotype"] == phenotype
    ]

    # Sort the cluster sizes from the biggest
    sizes = phenotype_data["size"].to_numpy(dtype=int)
    sizes_sorted = np.sort(sizes)[::-1]

    # Number of cells and number of clusters
    number_of_cells_phenotype = int(np.sum(sizes))
    number_of_clusters = int(sizes.size)

    # Take the 1st and 2nd largest clusters
    s1 = int(sizes_sorted[0]) if sizes_sorted.size >= 1 else 0
    s2 = int(sizes_sorted[1]) if sizes_sorted.size >= 2 else 0

    # Calculate the order parameter
    denominator = s1 + s2
    psi = (
        (s1 - s2) / denominator
        if denominator > 0
        else np.nan
    )

    # Isolated cells
    number_of_singletons = int(np.sum(sizes == 1))

    # Delete the giant cluster
    if sizes.size > 0:
        finite_sizes = np.delete(sizes, np.argmax(sizes))
    else:
        finite_sizes = np.array([], dtype=int)

    # Calculate the mean finite cluster size
    if finite_sizes.size > 0 and np.sum(finite_sizes) > 0:
        mean_finite_cluster_size = float(
            np.sum(finite_sizes**2) / np.sum(finite_sizes)
        )
        arithmetic_mean_without_largest = float(
            np.mean(finite_sizes)
        )
    else:
        mean_finite_cluster_size = np.nan
        arithmetic_mean_without_largest = np.nan

    # Calculate the total number of cells
    total_number_of_cells = int(raw_data["size"].sum())

    # Take s1 and s2 normalized
    if number_of_cells_phenotype > 0:
        s1_over_n_phenotype = s1 / number_of_cells_phenotype
        s2_over_n_phenotype = s2 / number_of_cells_phenotype
        isolated_fraction_phenotype = (
            number_of_singletons / number_of_cells_phenotype
        )
    else:
        s1_over_n_phenotype = np.nan
        s2_over_n_phenotype = np.nan
        isolated_fraction_phenotype = np.nan

    if total_number_of_cells > 0:
        s1_over_n = s1 / total_number_of_cells
        s2_over_n = s2 / total_number_of_cells
        fraction_phenotype = (
            number_of_cells_phenotype / total_number_of_cells
        )
        isolated_fraction_total = (
            number_of_singletons / total_number_of_cells
        )
    else:
        s1_over_n = np.nan
        s2_over_n = np.nan
        fraction_phenotype = np.nan
        isolated_fraction_total = np.nan

    return {
        **metadata,
        "phenotype": phenotype,
        "total_number_of_cells": total_number_of_cells,
        "number_of_cells_phenotype": number_of_cells_phenotype,
        "fraction_phenotype": fraction_phenotype,
        "number_of_clusters": number_of_clusters,
        "S1": s1,
        "S2": s2,
        "S1_over_N": s1_over_n,
        "S2_over_N": s2_over_n,
        "S1_over_Nphenotype": s1_over_n_phenotype,
        "S2_over_Nphenotype": s2_over_n_phenotype,
        "psi": psi,
        "number_of_singletons": number_of_singletons,
        "isolated_fraction_total": isolated_fraction_total,
        "isolated_fraction_phenotype": isolated_fraction_phenotype,
        "mean_finite_cluster_size": mean_finite_cluster_size,
        "arithmetic_mean_without_largest": arithmetic_mean_without_largest,
    }


# Create a function to create the cluster size distribution
def calculate_size_counts(
    raw_data: pd.DataFrame,
    metadata: dict,
) -> pd.DataFrame:
    """Count how many clusters of each size are present in one snapshot."""
    counts = (
        raw_data.groupby(["phenotype", "size"])
        .size()
        .rename("count")
        .reset_index()
    )

    return add_metadata(counts, metadata)

# Create a function to process raw files into one
def process_raw_files(
    raw_files: list[Path],
    raw_output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process raw cluster files and stream them into one parquet file."""
    snapshot_rows = []
    size_count_tables = []
    writer = None

    if raw_output_path.exists():
        raw_output_path.unlink()

    try:
        for index, filepath in enumerate(raw_files, start=1):
            metadata = parse_metadata(filepath.name)
            raw_data = pd.read_csv(filepath, skipinitialspace=True)

            for phenotype in ("round", "elongated"):
                snapshot_rows.append(
                    calculate_snapshot_statistics(
                        raw_data=raw_data,
                        metadata=metadata,
                        phenotype=phenotype,
                    )
                )

            size_count_tables.append(
                calculate_size_counts(
                    raw_data=raw_data,
                    metadata=metadata,
                )
            )

            raw_with_metadata = add_metadata(
                raw_data,
                metadata,
            )

            table = pa.Table.from_pandas(
                raw_with_metadata,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    raw_output_path,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)

            if index % 500 == 0 or index == len(raw_files):
                print(
                    f"Raw cluster files processed: "
                    f"{index}/{len(raw_files)}"
                )
    finally:
        if writer is not None:
            writer.close()

    snapshots = pd.DataFrame(snapshot_rows)

    if size_count_tables:
        size_counts = pd.concat(
            size_count_tables,
            ignore_index=True,
        )
    else:
        size_counts = pd.DataFrame()

    return snapshots, size_counts

# Create a function to process summary files into one
def process_summary_files(
    summary_files: list[Path],
) -> pd.DataFrame:
    """Combine all compact cluster summaries into one time series."""
    tables = []

    for index, filepath in enumerate(summary_files, start=1):
        metadata = parse_metadata(filepath.name)
        summary = pd.read_csv(filepath, skipinitialspace=True)

        total_number_of_cells = int(
            summary["total_number_of_cells"].sum()
        )

        summary["S1"] = summary["largest_cluster_size"].fillna(0)
        summary["fraction_phenotype"] = (
            summary["total_number_of_cells"]
            / total_number_of_cells
        )
        summary["S1_over_N"] = (
            summary["S1"] / total_number_of_cells
        )
        summary["S1_over_Nphenotype"] = np.where(
            summary["total_number_of_cells"] > 0,
            summary["S1"] / summary["total_number_of_cells"],
            np.nan,
        )

        tables.append(
            add_metadata(
                summary,
                metadata,
            )
        )

        if index % 1000 == 0 or index == len(summary_files):
            print(
                f"Summary cluster files processed: "
                f"{index}/{len(summary_files)}"
            )

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True)

# Create a function to merge summary and raw files for cluster time series
def merge_raw_observables_into_time_series(
    time_series: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Add S2 and raw-derived observables at the available raw timesteps."""
    if time_series.empty or snapshots.empty:
        return time_series

    merge_keys = [
        "N",
        "rho",
        "seed",
        "step",
        "phenotype",
        "initial_fraction_elongated",
        "force",
    ]

    raw_columns = merge_keys + [
        "S2",
        "S2_over_N",
        "S2_over_Nphenotype",
        "psi",
        "number_of_singletons",
        "isolated_fraction_total",
        "isolated_fraction_phenotype",
        "mean_finite_cluster_size",
    ]

    return time_series.merge(
        snapshots[raw_columns],
        on=merge_keys,
        how="left",
    )

# Finally, the main function
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate cluster output files into parquet datasets."
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
            "<data_root>/processed/clusters"
        ),
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_root / "processed" / "clusters"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_files = sorted(
        data_root.rglob("cluster_summary_*.dat")
    )
    raw_files = sorted(
        data_root.rglob("cluster_sizes_*.dat")
    )

    print(f"Summary files found: {len(summary_files)}")
    print(f"Raw files found: {len(raw_files)}")

    if not summary_files and not raw_files:
        raise FileNotFoundError(
            f"No cluster files found below {data_root}"
        )

    raw_output_path = output_dir / "cluster_raw.parquet"

    if raw_files:
        snapshots, size_counts = process_raw_files(
            raw_files=raw_files,
            raw_output_path=raw_output_path,
        )
    else:
        snapshots = pd.DataFrame()
        size_counts = pd.DataFrame()

    time_series = process_summary_files(summary_files)
    time_series = merge_raw_observables_into_time_series(
        time_series=time_series,
        snapshots=snapshots,
    )

    sort_columns = [
        "N",
        "rho",
        "seed",
        "step",
        "phenotype",
    ]

    if not time_series.empty:
        time_series = time_series.sort_values(
            sort_columns
        ).reset_index(drop=True)
        time_series.to_parquet(
            output_dir / "cluster_time_series.parquet",
            index=False,
        )

    if not snapshots.empty:
        snapshots = snapshots.sort_values(
            sort_columns
        ).reset_index(drop=True)
        snapshots.to_parquet(
            output_dir / "cluster_snapshots.parquet",
            index=False,
        )

    if not size_counts.empty:
        size_counts = size_counts.sort_values(
            sort_columns + ["size"]
        ).reset_index(drop=True)
        size_counts.to_parquet(
            output_dir / "cluster_size_counts.parquet",
            index=False,
        )

    print(f"Processed cluster data written to: {output_dir}")


if __name__ == "__main__":
    main()
