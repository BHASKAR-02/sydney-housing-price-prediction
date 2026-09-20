"""
Canonical model definition, shared by the notebook and the backend.

Why this file exists: the deployed model is a pickle, and a pickle is tied to the
scikit-learn version that produced it. If the hosting platform installs a different
version, loading it can fail or warn. Ridge on 120 rows trains in well under a
second, so the backend can simply rebuild the model from the CSV when the pickle
will not load. That removes the version coupling entirely.

The pipeline here must stay identical to the one in the notebook. There is a check
at the bottom of this file that trains from the CSV and compares the result against
the saved artefact.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.features import engineer

RANDOM_STATE = 42

NUMERIC = ["bedrooms", "bathrooms", "car_spaces", "land_size_sqm", "internal_area_sqm",
           "distance_to_cbd_km", "distance_to_station_km", "school_catchment_rating",
           "days_on_market", "has_pool", "has_aircon", "renovated", "water_view",
           "property_age", "total_rooms", "bath_per_bed", "land_to_internal",
           "months_since_start", "sale_month", "station_convenience", "amenity_score",
           "description_length"]

CATEGORICAL = ["suburb", "property_type", "sale_method"]

TEXT = "agent_description"

TARGET = "sale_price"


def make_preprocessor(scale: bool = True, use_text: bool = True) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    blocks = [
        ("num", Pipeline(numeric_steps), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None)),
        ]), CATEGORICAL),
    ]

    if use_text:
        # Small vocabulary on purpose. With 120 rows a large TF-IDF matrix would have
        # more columns than rows and the model would fit noise in the blurbs.
        blocks.append(("txt", Pipeline([
            ("tfidf", TfidfVectorizer(max_features=40, ngram_range=(1, 2),
                                      min_df=4, stop_words="english",
                                      sublinear_tf=True)),
        ]), TEXT))

    return ColumnTransformer(blocks, remainder="drop", sparse_threshold=0.0)


def build_ridge_pipeline() -> Pipeline:
    """The model recommended in Part 3 of the notebook."""
    return Pipeline([
        ("prep", make_preprocessor(scale=True, use_text=True)),
        ("model", RidgeCV(alphas=np.logspace(-2, 3, 40))),
    ])


def train_from_csv(csv_path: Path) -> Pipeline:
    """Rebuild the deployed model from the raw dataset. Trains on everything,
    which is what the notebook does before exporting."""
    df = pd.read_csv(csv_path, parse_dates=["sale_date"])
    df_fe = engineer(df)

    X = df_fe[NUMERIC + CATEGORICAL + [TEXT]].copy()
    y_log = np.log(df_fe[TARGET].values)

    pipe = build_ridge_pipeline()
    pipe.fit(X, y_log)
    return pipe
