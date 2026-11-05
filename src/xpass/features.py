
from __future__ import annotations

import ast
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Model feature specification
# --------------------------------------------------

MODEL_FEATURES = [
    "x",
    "y",
    "end_x",
    "end_y",
    "pass_length",
    "pass_angle_sin",
    "pass_angle_cos",
    "is_head_pass",
    "is_cross",
    "is_set_piece",
    "is_chipped",
    "passes_in_possession",
    "previous_action_same_team",
    "previous_pass_length",
    "previous_pass_angle_sin",
    "previous_pass_angle_cos"
]


REQUIRED_COLUMNS = [
    "game_id",
    "period",
    "minute",
    "second",
    "type",
    "team",
    "outcome_type",
    "qualifiers",
    "x",
    "y",
    "end_x",
    "end_y"
]


# --------------------------------------------------
# 2. Extract pass features from qualifiers
# --------------------------------------------------

def extract_pass_features(qualifiers):
    """
    Extract relevant passing information from a WhoScored
    qualifiers field.
    """

    features = {
        "pass_length": np.nan,
        "pass_angle": np.nan,
        "is_chipped": 0,
        "is_head_pass": 0,
        "is_cross": 0,
        "is_freekick": 0,
        "is_goal_kick": 0,
        "is_corner": 0,
        "is_indirect_fk": 0
    }

    if isinstance(qualifiers, str):
        try:
            qualifiers = ast.literal_eval(qualifiers)
        except (ValueError, SyntaxError):
            return features

    if not isinstance(qualifiers, list):
        return features

    for q in qualifiers:

        if not isinstance(q, dict):
            continue

        qualifier_type = q.get("type")

        if not isinstance(qualifier_type, dict):
            continue

        name = qualifier_type.get("displayName")
        value = q.get("value")

        if name == "Length":
            features["pass_length"] = value

        elif name == "Angle":
            features["pass_angle"] = value

        elif name == "Chipped":
            features["is_chipped"] = 1

        elif name == "HeadPass":
            features["is_head_pass"] = 1

        elif name == "Cross":
            features["is_cross"] = 1

        elif name == "FreekickTaken":
            features["is_freekick"] = 1

        elif name == "GoalKick":
            features["is_goal_kick"] = 1

        elif name == "CornerTaken":
            features["is_corner"] = 1

        elif name == "IndirectFreekickTaken":
            features["is_indirect_fk"] = 1

    return features


# --------------------------------------------------
# 3. Assign possessions to events
# --------------------------------------------------

def assign_possessions(match_events):
    """
    Assign possession IDs to events from a single match.

    Events must already be in chronological order.
    """

    df = match_events.copy()

    control_events = {
        "Pass",
        "BallRecovery",
        "Interception",
        "Clearance",
        "BallTouch",
        "TakeOn",
        "Claim",
        "BlockedPass",
        "KeeperPickup",
        "PenaltyFaced",
        "CrossNotClaimed",
        "Smother",
        "GoodSkill",
        "ChanceMissed"
    }

    possession_ids = []

    possession_id = 0
    possession_team = None

    for _, event in df.iterrows():

        event_type = event["type"]
        team = event["team"]
        outcome = event["outcome_type"]

        # Aerials are contested events
        if event_type == "Aerial":
            possession_ids.append(pd.NA)
            continue

        # Fouls
        if event_type == "Foul":

            if outcome == "Successful":
                possession_id += 1
                possession_team = team
                possession_ids.append(possession_id)

            else:
                possession_ids.append(pd.NA)

            continue

        # Offside stoppages
        if event_type == "OffsideGiven":
            possession_ids.append(pd.NA)
            possession_team = None
            continue

        if event_type in {"OffsidePass", "OffsideProvoked"}:
            possession_ids.append(pd.NA)
            continue

        # Corner awards
        if event_type == "CornerAwarded":

            if outcome == "Successful":
                possession_id += 1
                possession_team = team
                possession_ids.append(possession_id)

            else:
                possession_ids.append(pd.NA)

            continue

        # Events that establish or continue control
        if event_type in control_events:

            if possession_team != team:
                possession_id += 1
                possession_team = team

            possession_ids.append(possession_id)
            continue

        # Tackles
        if event_type == "Tackle":

            possession_ids.append(pd.NA)

            if outcome == "Successful":
                possession_team = None

            continue

        # Goals
        if event_type == "Goal":

            if possession_team != team:
                possession_id += 1
                possession_team = team

            possession_ids.append(possession_id)
            possession_team = None

            continue

        # Events that do not directly establish possession
        if event_type in {
            "Challenge",
            "Save",
            "SubstitutionOn",
            "SubstitutionOff",
            "End",
            "Dispossessed",
            "Card",
            "KeeperSweeper",
            "Punch",
            "FormationSet",
            "FormationChange",
            "ShieldBallOpp",
            "Error"
        }:

            possession_ids.append(pd.NA)
            continue

        # Shots
        if event_type in {"MissedShots", "SavedShot", "ShotOnPost"}:

            if possession_team != team:
                possession_id += 1
                possession_team = team

            possession_ids.append(possession_id)
            continue

        # Start of a period
        if event_type == "Start":

            possession_ids.append(pd.NA)
            possession_team = None
            continue

        # Other events inherit possession where possible
        if possession_team is not None:
            possession_ids.append(possession_id)
        else:
            possession_ids.append(pd.NA)

    df["possession_id"] = pd.array(
        possession_ids,
        dtype="Int64"
    )

    return df


# --------------------------------------------------
# 4. Create contextual features
# --------------------------------------------------

def create_contextual_features(events):
    """
    Calculate possession counts and previous-event features
    for every pass, using the complete event sequence.
    """

    events = events.copy()

    # Sort chronologically within each match.
    # _original_order breaks ties when timestamps are identical.
    events["_sort_minute"] = pd.to_numeric(
        events["minute"], errors="coerce"
    )

    events["_sort_second"] = pd.to_numeric(
        events["second"], errors="coerce"
    )

    ordered = events.sort_values(
        [
            "game_id",
            "period",
            "_sort_minute",
            "_sort_second",
            "_original_order"
        ],
        na_position="last",
        kind="stable"
    ).copy()

    # The explicit loop keeps possession assignment independent
    # between matches and avoids relying on groupby.apply indexing.
    ordered["possession_id"] = pd.Series(
        pd.NA, index=ordered.index, dtype="Int64"
    )

    for game_id, match_events in ordered.groupby(
        "game_id", sort=False
    ):

        assigned = assign_possessions(match_events)

        ordered.loc[
            assigned.index, "possession_id"
        ] = assigned["possession_id"]

    # Passes in the current possession
    ordered["passes_in_possession"] = np.nan

    pass_mask = ordered["type"].eq("Pass")

    ordered.loc[pass_mask, "passes_in_possession"] = (
        ordered.loc[pass_mask]
        .groupby(
            ["game_id", "possession_id"],
            dropna=True,
            sort=False
        )
        .cumcount()
    )

    # Previous event information
    previous_columns = [
        "end_x",
        "end_y",
        "possession_id",
        "team",
        "type",
        "pass_length",
        "pass_angle"
    ]

    for column in previous_columns:

        ordered[f"previous_{column}"] = (
            ordered.groupby("game_id", sort=False)[column]
            .shift(1)
        )

    # Previous action was by the same team
    ordered["previous_action_same_team"] = np.where(
        ordered["previous_team"].isna(),
        np.nan,
        (
            ordered["team"] == ordered["previous_team"]
        ).astype(int)
    )

    # Previous pass length and angle
    ordered["previous_pass_length"] = pd.to_numeric(
        ordered["previous_pass_length"],
        errors="coerce"
    )

    ordered["previous_pass_angle"] = pd.to_numeric(
        ordered["previous_pass_angle"],
        errors="coerce"
    )

    # Convert angles into sine and cosine
    ordered["pass_angle"] = pd.to_numeric(
        ordered["pass_angle"],
        errors="coerce"
    )

    ordered["pass_angle_sin"] = np.sin(
        ordered["pass_angle"]
    )

    ordered["pass_angle_cos"] = np.cos(
        ordered["pass_angle"]
    )

    ordered["previous_pass_angle_sin"] = np.sin(
        ordered["previous_pass_angle"]
    )

    ordered["previous_pass_angle_cos"] = np.cos(
        ordered["previous_pass_angle"]
    )

    # Return only pass events, restoring their original input order
    passes = ordered.loc[pass_mask].sort_values(
        "_original_order",
        kind="stable"
    ).copy()

    return passes


# --------------------------------------------------
# 5. Main feature-engineering function
# --------------------------------------------------

def create_xpass_features(events):
    """
    Convert raw WhoScored event data into a pass-level
    dataframe containing the features required by the xPass model.

    Accepts one match or multiple matches.
    """

    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in events.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required event columns: {missing_columns}"
        )

    if events.empty:
        raise ValueError("The supplied event dataframe is empty.")

    # Work with a clean, unique positional index
    events = events.copy().reset_index(drop=True)

    # Preserve original event order
    if "_original_order" not in events.columns:
        events["_original_order"] = np.arange(len(events))

    # Extract pass events
    pass_mask = events["type"].eq("Pass")

    passes = events.loc[pass_mask].copy()

    if passes.empty:
        raise ValueError("No pass events were found.")

    # Extract qualifiers for all passes
    feature_data = passes["qualifiers"].apply(
        extract_pass_features
    )

    feature_df = pd.DataFrame(
        feature_data.tolist(),
        index=passes.index
    )

    passes = pd.concat(
        [passes, feature_df],
        axis=1
    )

    # Convert numeric qualifier values
    for column in ["pass_length", "pass_angle"]:
        passes[column] = pd.to_numeric(
            passes[column],
            errors="coerce"
        )

    # Set-piece and open-play flags
    set_piece_features = [
        "is_freekick",
        "is_goal_kick",
        "is_corner",
        "is_indirect_fk"
    ]

    passes["is_set_piece"] = (
        passes[set_piece_features].max(axis=1)
    )

    passes["is_open_play"] = (
        1 - passes["is_set_piece"]
    )

    # Put extracted pass features onto the complete event data
    events = events.join(
        passes[
            [
                "pass_length",
                "pass_angle",
                "is_chipped",
                "is_head_pass",
                "is_cross",
                "is_freekick",
                "is_goal_kick",
                "is_corner",
                "is_indirect_fk",
                "is_set_piece",
                "is_open_play"
            ]
        ],
        how="left"
    )

    # Generate contextual features using all events
    passes = create_contextual_features(events)

    # Convert pass angles into circular features
    passes["pass_angle_sin"] = np.sin(
        pd.to_numeric(passes["pass_angle"], errors="coerce")
    )

    passes["pass_angle_cos"] = np.cos(
        pd.to_numeric(passes["pass_angle"], errors="coerce")
    )

    passes["previous_pass_angle_sin"] = np.sin(
        pd.to_numeric(
            passes["previous_pass_angle"], errors="coerce"
        )
    )

    passes["previous_pass_angle_cos"] = np.cos(
        pd.to_numeric(
            passes["previous_pass_angle"], errors="coerce"
        )
    )

    # Create binary target
    passes["outcome"] = (
        passes["outcome_type"] == "Successful"
    ).astype(int)

    # Metadata to retain alongside the model features
    metadata_columns = [
        "league",
        "season",
        "game_id",
        "player_id",
        "player",
        "team",
        "period",
        "minute",
        "second",
        "possession_id",
        "outcome_type",
        "outcome"
    ]

    # Keep only metadata columns that exist in the input
    metadata_columns = [
        column for column in metadata_columns
        if column in passes.columns
    ]

    output_columns = list(dict.fromkeys(
        MODEL_FEATURES + metadata_columns
    ))

    result = passes[output_columns].copy()

    return result
