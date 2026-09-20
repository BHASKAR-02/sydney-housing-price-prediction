"""
Deployment smoke test.

Run this in an environment that has ONLY the runtime requirements installed, which
is what the hosting platform gives you:

    python -m venv /tmp/leanenv
    /tmp/leanenv/bin/pip install -r requirements.txt
    /tmp/leanenv/bin/python scripts/smoke_test.py

It exercises every code path the deployed app uses and fails loudly if any of them
needs something that is not in requirements.txt. Two real deployment bugs would
have been caught here rather than on Streamlit Cloud:

  1. matplotlib missing, because the model card used a pandas Styler method
     (background_gradient) that quietly depends on it
  2. the saved pickle refusing to load under a different scikit-learn version

Add --break-model to simulate the second one.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BREAK = "--break-model" in sys.argv
failures = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        failures.append(name)


backup = None
model_path = ROOT / "models" / "best_model.joblib"
if BREAK:
    backup = model_path.read_bytes()
    # a valid pickle whose global lookup fails, the way a version mismatch does
    model_path.write_bytes(
        b"csklearn.compose._column_transformer\n_RemainderColsListMISSING\n.")
    print("simulating an unloadable model file\n")

try:
    import pandas as pd
    from backend import main as api

    print("checking deployed code paths")

    api.load_model()
    check("model available", lambda: (_ for _ in ()).throw(
        RuntimeError("model is None")) if api.MODEL is None else None)
    print(f"        source: {api.MODEL_SOURCE}")

    check("metadata endpoint", lambda: api.metadata()["model_name"])

    check("single prediction", lambda: api.predict_one(api.PropertyIn(
        suburb="Mosman", property_type="House", bedrooms=4, bathrooms=3,
        land_size_sqm=600, internal_area_sqm=250, water_view=1, renovated=1)))

    def batch():
        required = ["suburb", "property_type", "bedrooms", "bathrooms",
                    "internal_area_sqm"]
        defaults = {"car_spaces": 1, "land_size_sqm": 0.0, "year_built": 1995,
                    "has_pool": 0, "has_aircon": 1, "renovated": 0, "water_view": 0,
                    "distance_to_station_km": 1.5, "sale_method": "Auction",
                    "days_on_market": 30, "sale_date": "2025-08-01",
                    "agent_description": ""}
        raw = pd.DataFrame([
            {"suburb": "Mosman", "property_type": "House", "bedrooms": 5,
             "bathrooms": 3, "internal_area_sqm": 300},
            {"suburb": "Blacktown", "property_type": "Townhouse", "bedrooms": 3,
             "bathrooms": 2, "internal_area_sqm": 145}])
        for c, v in defaults.items():
            raw[c] = v if c not in raw.columns else raw[c]
        for rec in raw.to_dict("records"):
            payload = {k: rec[k] for k in list(defaults) + required if k in rec}
            payload["sale_date"] = str(payload["sale_date"])[:10]
            payload["agent_description"] = str(payload.get("agent_description") or "")
            api.predict_one(api.PropertyIn(**payload))
    check("batch prediction", batch)

    def styler():
        """The model card table. Catches matplotlib-dependent Styler methods."""
        meta = api.metadata()
        m = (pd.DataFrame(meta["all_model_metrics"]).T[["MAE", "RMSE", "R2", "MAPE"]]
             .sort_values("MAE"))

        def hw(row):
            best = row.name == m.index[0]
            return ["background-color: #3B5BA51A; font-weight: 600" if best else ""] * len(row)

        m.style.format({"MAE": "${:,.0f}", "RMSE": "${:,.0f}",
                        "R2": "{:.3f}", "MAPE": "{:.1f}%"}).apply(hw, axis=1).to_html()
    check("model card styling", styler)

    def warnings_fire():
        r = api.predict_one(api.PropertyIn(
            suburb="Mosman", property_type="Apartment", bedrooms=1, bathrooms=1,
            internal_area_sqm=52))
        assert any("Small dwellings" in w for w in r.warnings), "expected warning missing"
    check("risk warnings", warnings_fire)

finally:
    if backup is not None:
        model_path.write_bytes(backup)
        print("\nrestored the real model file")

print()
if failures:
    print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("all deployment checks passed")
