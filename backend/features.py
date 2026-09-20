"""
Feature engineering shared with the notebook.

This is a copy of the engineer() function from notebooks/sydney_housing_analysis.ipynb.
It has to match exactly, because the saved pipeline expects the engineered columns to
already be there. If you change one you must change the other, and there is a test at
the bottom of this file that checks the column list still lines up with what the model
was trained on.
"""

import pandas as pd

# The reference date used for months_since_start. This is the first sale date in the
# training data. It must not change, otherwise the feature means something different
# at prediction time than it did at training time.
REFERENCE_DATE = pd.Timestamp("2023-09-01")


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()

    if not pd.api.types.is_datetime64_any_dtype(d["sale_date"]):
        d["sale_date"] = pd.to_datetime(d["sale_date"])

    # An apartment has no land. That is a real zero, not a missing value.
    d.loc[d["property_type"] == "Apartment", "land_size_sqm"] = d.loc[
        d["property_type"] == "Apartment", "land_size_sqm"].fillna(0.0)

    d["property_age"] = d["sale_date"].dt.year - d["year_built"]
    d["total_rooms"] = d["bedrooms"] + d["bathrooms"]
    d["bath_per_bed"] = d["bathrooms"] / d["bedrooms"].clip(lower=1)
    d["land_to_internal"] = d["land_size_sqm"] / d["internal_area_sqm"].clip(lower=1)
    d["months_since_start"] = ((d["sale_date"] - REFERENCE_DATE).dt.days / 30.44).round(1)
    d["sale_month"] = d["sale_date"].dt.month
    d["station_convenience"] = 1.0 / (1.0 + d["distance_to_station_km"])
    d["amenity_score"] = (d["has_pool"] + d["has_aircon"]
                          + d["renovated"] + d["water_view"])
    d["description_length"] = d["agent_description"].fillna("").str.split().str.len()

    return d
