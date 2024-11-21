"""
Data Preparation Module for NFL Big Data Bowl 2024

This module processes raw NFL tracking data to prepare it for machine learning models.
It includes functions for loading, cleaning, and transforming the data, as well as
splitting it into train, validation, and test sets.

Functions:
    get_players_df: Load and preprocess player data
    get_plays_df: Load and preprocess play data with coverage mapping
    get_tracking_df: Load and preprocess tracking data
    add_features_to_tracking_df: Add derived features to tracking data
    convert_tracking_to_cartesian: Convert polar coordinates to Cartesian
    standardize_tracking_directions: Standardize play directions
    augment_mirror_tracking: Augment data by mirroring the field
    get_coverage_target_df: Generate target dataframe for coverage classification
    split_train_test_val: Split data into train, validation, and test sets
    main: Main execution function

"""

from argparse import ArgumentParser
from pathlib import Path

import polars as pl

INPUT_DATA_DIR = Path("../data/")

# Remap dictionary for coverage types
REMAPPING_DICT = {
    'Cover-3 Double Cloud': 'Cover-3',
    'Miscellaneous': 'Misc',
    'Cover-3 Cloud Left': 'Cover-3',
    'Cover-3 Cloud Right': 'Cover-3',
    'Prevent': 'Prevent',
    'Cover-1 Double': 'Cover-1',
    'Bracket': 'Bracket',
    'Goal Line': 'Goal Line',
    '2-Man': '2-Man',
    'NA': 'Misc',
    'Red Zone': 'Red Zone',
    'Cover-0': 'Cover-0',
    'Cover-3 Seam': 'Cover-3',
    'Cover-6 Right': 'Cover-6',
    'Cover 6-Left': 'Cover-6',
    'Cover-2': 'Cover-2',
    'Quarters': 'Quarters',
    'Cover-1': 'Cover-1',
    'Cover-3': 'Cover-3'
}


def get_players_df() -> pl.DataFrame:
    """
    Load player-level data and preprocesses features.

    Returns:
        pl.DataFrame: Preprocessed player data with additional features.
    """
    return (
        pl.read_csv(INPUT_DATA_DIR / "players.csv", null_values=["NA", "nan", "N/A", "NaN", ""])
        .with_columns(
            height_inches=(
                pl.col("height").str.split("-").map_elements(lambda s: int(s[0]) * 12 + int(s[1]), return_dtype=int)
            )
        )
        .with_columns(
            weight_Z=(pl.col("weight") - pl.col("weight").mean()) / pl.col("weight").std(),
            height_Z=(pl.col("height_inches") - pl.col("height_inches").mean()) / pl.col("height_inches").std(),
        )
    )


def get_plays_df() -> pl.DataFrame:
    """
    Load play-level data, preprocess features, and map coverage types.

    Returns:
        pl.DataFrame: Preprocessed play data with additional features and remapped coverage types.
    """
    plays_df = pl.read_csv(INPUT_DATA_DIR / "plays.csv", null_values=["NA", "nan", "N/A", "NaN", ""]).with_columns(
        distanceToGoal=(
            pl.when(pl.col("possessionTeam") == pl.col("yardlineSide"))
            .then(100 - pl.col("yardlineNumber"))
            .otherwise(pl.col("yardlineNumber"))
        )
    )
    
    # Remap pff_passCoverage using the provided dictionary
    plays_df = plays_df.with_columns(
        coverage=pl.col("pff_passCoverage").replace(REMAPPING_DICT).fill_null("Misc")
    )
    
    return plays_df


def get_tracking_df() -> pl.DataFrame:
    """
    Load tracking data and preprocesses features. Notably, exclude rows representing the football's movement.

    Returns:
        pl.DataFrame: Preprocessed tracking data with additional features.
    """
    # don't include football rows for this project
    return pl.read_csv(INPUT_DATA_DIR / "tracking_week_*.csv", null_values=["NA", "nan", "N/A", "NaN", ""]).filter(
        pl.col("displayName") != "football"
    )


def add_features_to_tracking_df(
    tracking_df: pl.DataFrame,
    players_df: pl.DataFrame,
    plays_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Consolidates play and player level data into the tracking data.

    Args:
        tracking_df (pl.DataFrame): Tracking data
        players_df (pl.DataFrame): Player data
        plays_df (pl.DataFrame): Play data

    Returns:
        pl.DataFrame: Tracking data with additional features.
    """
    og_len = len(tracking_df)
    tracking_df = (
        tracking_df.join(
            plays_df.select(
                "gameId",
                "playId",
                "coverage",
                "down",
                "possessionTeam",
                "yardsToGo",
                "distanceToGoal"
            ),
            on=["gameId", "playId"],
            how="inner",
        )
        .join(
            players_df.select(["nflId", "weight_Z", "height_Z"]).unique(),
            on="nflId",
            how="inner",
        )
        .with_columns(
            side=pl.when(pl.col("club") == pl.col("possessionTeam"))
            .then(pl.lit(1))
            .otherwise(pl.lit(-1))
            .alias("side"),
        )
        .drop(["possessionTeam"])
    )
    assert len(tracking_df) == og_len, "Lost rows when joining tracking data with play/player data"

    return tracking_df


def convert_tracking_to_cartesian(tracking_df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert polar coordinates to Unit-circle Cartesian format.

    Args:
        tracking_df (pl.DataFrame): Tracking data

    Returns:
        pl.DataFrame: Tracking data with Cartesian coordinates.
    """
    return (
        tracking_df.with_columns(
            dir=((pl.col("dir") - 90) * -1) % 360,
            o=((pl.col("o") - 90) * -1) % 360,
        )
        # convert polar vectors to cartesian ((s, dir) -> (vx, vy), (o) -> (ox, oy))
        .with_columns(
            vx=pl.col("s") * pl.col("dir").radians().cos(),
            vy=pl.col("s") * pl.col("dir").radians().sin(),
            ox=pl.col("o").radians().cos(),
            oy=pl.col("o").radians().sin(),
        )
    )


def standardize_tracking_directions(tracking_df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize play directions to always moving left to right.

    Args:
        tracking_df (pl.DataFrame): Tracking data

    Returns:
        pl.DataFrame: Tracking data with standardized directions.
    """
    return tracking_df.with_columns(
        x=pl.when(pl.col("playDirection") == "right").then(pl.col("x")).otherwise(120 - pl.col("x")),
        y=pl.when(pl.col("playDirection") == "right").then(pl.col("y")).otherwise(53.3 - pl.col("y")),
        vx=pl.when(pl.col("playDirection") == "right").then(pl.col("vx")).otherwise(-1 * pl.col("vx")),
        vy=pl.when(pl.col("playDirection") == "right").then(pl.col("vy")).otherwise(-1 * pl.col("vy")),
        ox=pl.when(pl.col("playDirection") == "right").then(pl.col("ox")).otherwise(-1 * pl.col("ox")),
        oy=pl.when(pl.col("playDirection") == "right").then(pl.col("oy")).otherwise(-1 * pl.col("oy")),
    ).drop("playDirection")


def augment_mirror_tracking(tracking_df: pl.DataFrame) -> pl.DataFrame:
    """
    Augment data by mirroring the field assuming all plays are moving right.
    There are arguments to not do this as football isn't perfectly symmetric (e.g. most QBs are right-handed) but
    tackling is mostly symmetrical and for the sake of this demo I think more data is more important.

    Args:
        tracking_df (pl.DataFrame): Tracking data

    Returns:
        pl.DataFrame: Augmented tracking data.
    """
    og_len = len(tracking_df)

    mirrored_tracking_df = tracking_df.clone().with_columns(
        # only flip y values
        y=53.3 - pl.col("y"),
        vy=-1 * pl.col("vy"),
        oy=-1 * pl.col("oy"),
        mirrored=pl.lit(True),
    )

    tracking_df = pl.concat(
        [
            tracking_df.with_columns(mirrored=pl.lit(False)),
            mirrored_tracking_df,
        ],
        how="vertical",
    )

    assert len(tracking_df) == og_len * 2, "Lost rows when mirroring tracking data"
    return tracking_df


def get_coverage_target_df(tracking_df: pl.DataFrame, plays_df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate target dataframe for defensive coverage classification.

    Args:
        tracking_df (pl.DataFrame): Tracking data
        plays_df (pl.DataFrame): Play data

    Returns:
        pl.DataFrame: Target dataframe with coverage labels.
    """
    # Select unique play identifiers with mirrored from tracking_df
    unique_play_mirrored = tracking_df.select(["gameId", "playId", "mirrored"]).unique()

    # Join with plays_df to get coverage
    coverage_df = unique_play_mirrored.join(
        plays_df.select(["gameId", "playId", "coverage"]),
        on=["gameId", "playId"],
        how="left"  # Use left join to retain all mirrored plays
    ).with_columns(
        coverage=pl.col("coverage").fill_null("Misc")  # Handle any missing coverage
    )

    # Verify if any coverage is still null
    null_coverage_count = coverage_df.filter(pl.col("coverage").is_null()).height
    if null_coverage_count > 0:
        print(f"Warning: {null_coverage_count} plays have null coverage after join. Filling with 'Misc'.")
        coverage_df = coverage_df.with_columns(
            coverage=pl.col("coverage").fill_null("Misc")
        )

    # Ensure that tracking_df only includes plays present in coverage_df
    og_play_count = len(tracking_df.select(["gameId", "playId", "mirrored"]).unique())
    tracking_df = tracking_df.join(
        coverage_df.select(["gameId", "playId", "mirrored"]).unique(),
        on=["gameId", "playId", "mirrored"],
        how="inner",
    )
    new_play_count = len(tracking_df.select(["gameId", "playId", "mirrored"]).unique())
    print(f"Lost {(og_play_count - new_play_count)/og_play_count:.3%} plays when joining with coverage_df")

    return coverage_df


def split_train_test_val(tracking_df: pl.DataFrame, target_df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Split data into train, validation, and test sets.
    Split is 70-15-15 for train-test-val respectively. Notably, we split at the play level and not frame level.
    This ensures no target contamination between splits.

    Args:
        tracking_df (pl.DataFrame): Tracking data
        target_df (pl.DataFrame): Target data

    Returns:
        dict: Dictionary containing train, validation, and test dataframes.
    """
    tracking_df = tracking_df.sort(["gameId", "playId", "mirrored", "frameId"])
    target_df = target_df.sort(["gameId", "playId", "mirrored"])

    print(
        f"Total set: {tracking_df.n_unique(['gameId', 'playId', 'mirrored'])} plays,",
        f"{tracking_df.n_unique(['gameId', 'playId', 'mirrored', 'frameId'])} frames",
    )

    # Sample 30% of the plays for test and validation
    test_val_ids = target_df.select(["gameId", "playId", "mirrored"]).unique(maintain_order=True).sample(fraction=0.3, seed=42)
    train_tracking_df = tracking_df.join(test_val_ids, on=["gameId", "playId", "mirrored"], how="anti")
    train_tgt_df = target_df.join(test_val_ids, on=["gameId", "playId", "mirrored"], how="anti")
    print(
        f"Train set: {train_tracking_df.n_unique(['gameId', 'playId', 'mirrored'])} plays,",
        f"{train_tracking_df.n_unique(['gameId', 'playId', 'mirrored', 'frameId'])} frames",
    )

    # Split the 30% into test and validation (15% each)
    test_ids = test_val_ids.sample(fraction=0.5, seed=42)  # 15% for test
    test_tracking_df = tracking_df.join(test_ids, on=["gameId", "playId", "mirrored"], how="inner")
    test_tgt_df = target_df.join(test_ids, on=["gameId", "playId", "mirrored"], how="inner")
    print(
        f"Test set: {test_tracking_df.n_unique(['gameId', 'playId', 'mirrored'])} plays,",
        f"{test_tracking_df.n_unique(['gameId', 'playId', 'mirrored', 'frameId'])} frames",
    )

    val_ids = test_val_ids.join(test_ids, on=["gameId", "playId", "mirrored"], how="anti")
    val_tracking_df = tracking_df.join(val_ids, on=["gameId", "playId", "mirrored"], how="inner")
    val_tgt_df = target_df.join(val_ids, on=["gameId", "playId", "mirrored"], how="inner")
    print(
        f"Validation set: {val_tracking_df.n_unique(['gameId', 'playId', 'mirrored'])} plays,",
        f"{val_tracking_df.n_unique(['gameId', 'playId', 'mirrored', 'frameId'])} frames",
    )

    return {
        "train_features": train_tracking_df,
        "train_targets": train_tgt_df,
        "test_features": test_tracking_df,
        "test_targets": test_tgt_df,
        "val_features": val_tracking_df,
        "val_targets": val_tgt_df,
    }


def main():
    """
    Main execution function for data preparation.

    This function orchestrates the entire data preparation process, including:
    1. Loading raw data
    2. Adding features and transforming coordinates
    3. Generating target variables
    4. Splitting data into train, validation, and test sets
    5. Saving processed data to parquet files
    """
    players_df = get_players_df()
    plays_df = get_plays_df()
    tracking_df = get_tracking_df()

    tracking_df = add_features_to_tracking_df(tracking_df, players_df, plays_df)
    tracking_df = convert_tracking_to_cartesian(tracking_df)
    tracking_df = standardize_tracking_directions(tracking_df)
    tracking_df = augment_mirror_tracking(tracking_df)

    # Removed add_relative_positions as ball carrier is not needed

    coverage_tgt_df = get_coverage_target_df(tracking_df, plays_df)

    split_dfs = split_train_test_val(tracking_df, coverage_tgt_df)

    out_dir = Path("data/split_prepped_data/")
    out_dir.mkdir(exist_ok=True, parents=True)

    for key, df in split_dfs.items():
        if "targets" in key:
            # For target dataframes, no frame-level data is needed
            df = df.select(["gameId", "playId", "mirrored", "coverage"])
        else:
            # For feature dataframes, ensure proper sorting
            sort_keys = ["gameId", "playId", "mirrored", "frameId"]
            df = df.sort(sort_keys)
        df.write_parquet(out_dir / f"{key}.parquet")


if __name__ == "__main__":
    main()
