
from pathlib import Path
import joblib
import numpy as np

# --------------------------------------------------
# 1. Load model and feature list
# --------------------------------------------------

def load_xpass_model(model_path, features_path):
    """
    Load the trained xPass model and its required features.
    """

    model = joblib.load(model_path)
    features = joblib.load(features_path)

    return model, features


# --------------------------------------------------
# 2. Generate xPass predictions
# --------------------------------------------------

def predict_xpass(
    passes_df,
    model_path,
    features_path
):
    """
    Generate xPass probabilities and PAx for each pass.
    """

    # Load model and feature list
    model, features = load_xpass_model(
        model_path,
        features_path
    )

    # Check that all required features exist
    missing_features = [
        feature for feature in features
        if feature not in passes_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    # Select features in the exact training order
    X = passes_df[features].copy()

    # Identify the successful class
    successful_class = np.where(
        model.classes_ == 1
    )[0][0]

    # Predict probability of a successful pass
    passes_df = passes_df.copy()

    passes_df["xPass"] = model.predict_proba(X)[
        :, successful_class
    ]

    # Expected completion is the same probability,
    # expressed as an expected outcome for each pass.
    passes_df["expected_completion"] = passes_df["xPass"]

    # Calculate PAx where actual outcomes are available
    if "outcome" in passes_df.columns:

        passes_df["PAx"] = (
            passes_df["outcome"] - passes_df["xPass"]
        )

    return passes_df
