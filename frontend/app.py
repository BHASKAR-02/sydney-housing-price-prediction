"""
Streamlit front end for the Sydney housing price prediction tool.

It can run two ways:

  API mode        talks to the FastAPI backend over HTTP. This is the full
                  architecture and what I describe in the report.
  Standalone mode loads the model directly in process. I added this because the
                  free Streamlit Community Cloud tier only runs one process, so
                  without it I would need a second paid service for the API.

It picks API mode automatically if the backend is reachable, otherwise it falls
back to standalone. You can force either one in the sidebar.

Run:  streamlit run frontend/app.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Validated with the dataviz palette checker: passes lightness band, chroma floor,
# CVD separation (worst adjacent dE 24.0 deutan) and 3:1 contrast on a light surface.
BLUE, AMBER, PLUM = "#3B5BA5", "#B5770A", "#8B4A9C"
INK, MUTED, SURFACE, LINE = "#1F2933", "#6B7280", "#FCFCFB", "#E3E5E8"

st.set_page_config(page_title="Sydney Housing Price Estimator",
                   page_icon="house", layout="wide")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
  .hero {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:14px;
           padding:26px 30px; }}
  .hero-label {{ color:{MUTED}; font-size:.78rem; letter-spacing:.09em;
                 text-transform:uppercase; font-weight:600; }}
  .hero-value {{ color:{INK}; font-size:3.1rem; font-weight:700; line-height:1.05;
                 margin:6px 0 2px; font-variant-numeric:tabular-nums; }}
  .hero-sub {{ color:{MUTED}; font-size:.92rem; }}
  .flag {{ border-left:3px solid {AMBER}; background:#FDF8EF; padding:11px 14px;
           border-radius:0 8px 8px 0; margin-bottom:9px; font-size:.88rem;
           color:{INK}; }}
  .flag-ok {{ border-left-color:{BLUE}; background:#F2F5FB; }}
  .flag-title {{ font-weight:700; margin-right:6px; }}
  .band {{ height:11px; border-radius:6px; background:{LINE}; position:relative;
           margin:16px 0 7px; }}
  .band-fill {{ position:absolute; height:100%; border-radius:6px;
                background:{BLUE}; opacity:.28; }}
  .band-tick {{ position:absolute; width:3px; height:21px; top:-5px;
                background:{BLUE}; border-radius:2px; }}
  .band-ends {{ display:flex; justify-content:space-between; color:{MUTED};
                font-size:.8rem; font-variant-numeric:tabular-nums; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# backend access, either over HTTP or in process
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_local():
    """Standalone mode.

    This delegates to the backend's own startup routine rather than loading the
    pickle itself. That matters: the saved model is tied to the scikit-learn
    version that wrote it, and a host running a newer version fails with
    "module 'sklearn.compose._column_transformer' has no attribute
    '_RemainderColsList'". api.load_model() catches that and rebuilds the pipeline
    from the CSV, so routing through it means the standalone path gets the same
    protection the API path has.
    """
    from backend import main as api

    api.load_model()
    if api.MODEL is None:
        raise RuntimeError(
            "could not load models/best_model.joblib and could not rebuild the "
            "model from data/sydney_housing_raw.csv either")
    return api


def api_reachable() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=2).status_code == 200
    except Exception:
        return False


def get_metadata(mode):
    if mode == "API":
        return requests.get(f"{API_URL}/metadata", timeout=10).json()
    api = load_local()
    return api.metadata()


def call_predict(payload, mode):
    if mode == "API":
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
        if r.status_code != 200:
            st.error(f"API returned {r.status_code}: {r.text[:300]}")
            return None
        return r.json()
    api = load_local()
    return api.predict_one(api.PropertyIn(**payload)).model_dump()


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    detected = "API" if api_reachable() else "Standalone"
    choice = st.radio("Prediction backend",
                      ["Auto", "API", "Standalone"], index=0,
                      help="Auto uses the FastAPI backend when it is running.")
    MODE = detected if choice == "Auto" else choice

    if MODE == "API":
        st.success(f"Using FastAPI at {API_URL}")
    else:
        st.info("Running the model in process")
        if choice == "Auto":
            st.caption("Backend not reachable, fell back to standalone.")

    st.divider()
    try:
        META = get_metadata(MODE)
        st.markdown(
            f"**Deployed model**  \n{META['model_name']}  \n\n"
            f"**Cross validated error**  \n"
            f"${META['metrics']['MAE']:,.0f} typical miss  \n"
            f"{META['metrics']['MAPE']:.1f}% of sale price  \n"
            f"R squared {META['metrics']['R2']:.3f}")

        if MODE != "API":
            src = getattr(load_local(), "MODEL_SOURCE", "")
            if "retrained" in src:
                st.caption(
                    "The saved model file would not load under this version of "
                    "scikit-learn, so it was rebuilt from the dataset at startup. "
                    "Predictions are identical.")
        st.caption(f"Trained on {META['trained_on_rows']} sales across "
                   f"{len(META['suburbs'])} suburbs.")
    except Exception as e:
        st.error(f"Could not start the model: {e}")
        st.caption(
            "If this mentions scikit-learn or a pickle, the deployed library "
            "version differs from the one that trained the model. The app is "
            "meant to rebuild it from data/sydney_housing_raw.csv automatically, "
            "so check that file is present in the repository.")
        st.stop()

st.title("Sydney Housing Price Estimator")
st.caption("SIT720 task 8.1D. A decision support tool for Mosman, Parramatta and "
           "Blacktown. Every estimate comes with a confidence range and its known "
           "failure modes, because the model is wrong in predictable ways.")

tab_single, tab_batch, tab_model = st.tabs(
    ["Estimate one property", "Upload a CSV", "About the model"])


def render_result(res, suburb):
    """Prediction display. The headline is a stat tile rather than a chart, because
    a single number does not need axes. The range below it is the part people
    should actually read."""
    price = res["predicted_price"]
    lo, hi = res["lower_80"], res["upper_80"]

    st.markdown(f"""
    <div class="hero">
      <div class="hero-label">Estimated sale price</div>
      <div class="hero-value">${price:,.0f}</div>
      <div class="hero-sub">{res['confidence']} confidence &middot;
        80% of comparable sales fall within the range below</div>
    </div>""", unsafe_allow_html=True)

    span = hi - lo
    pos = (price - lo) / span * 100 if span > 0 else 50
    st.markdown(f"""
    <div class="band">
      <div class="band-fill" style="left:0%; width:100%"></div>
      <div class="band-tick" style="left:{pos:.1f}%"></div>
    </div>
    <div class="band-ends"><span>${lo:,.0f}</span>
      <span style="color:{INK};font-weight:600">estimate ${price:,.0f}</span>
      <span>${hi:,.0f}</span></div>
    """, unsafe_allow_html=True)

    st.caption(
        f"Range width {res['segment_uncertainty_pct']:.0f}% of the estimate. "
        "Wider ranges mean the model has seen fewer comparable sales for this "
        "suburb and property type, so this number is doing real work rather than "
        "being decoration.")

    med = res.get("comparable_median")
    if med:
        delta = (price - med) / med * 100
        st.caption(f"The median sale in {suburb} across the training data was "
                   f"${med:,.0f}. This estimate sits {delta:+.0f}% against it.")

    st.markdown("##### Before you use this number")
    ok = res["warnings"][0].startswith("No specific risk")
    for w in res["warnings"]:
        cls = "flag flag-ok" if ok else "flag"
        title = "Looks routine" if ok else "Caution"
        st.markdown(f'<div class="{cls}"><span class="flag-title">{title}:</span>'
                    f'{w}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# tab 1, single property
# ---------------------------------------------------------------------------
with tab_single:
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Property details")

        suburb = st.selectbox("Suburb", META["suburbs"],
                              index=META["suburbs"].index("Mosman")
                              if "Mosman" in META["suburbs"] else 0)
        ptype = st.selectbox("Property type", META["property_types"], index=1)

        c1, c2, c3 = st.columns(3)
        beds = c1.number_input("Bedrooms", 1, 10, 3)
        baths = c2.number_input("Bathrooms", 1, 8, 2)
        cars = c3.number_input("Car spaces", 0, 8, 1)

        internal = st.slider("Internal area (sqm)", 30, 800, 160, step=5)

        if ptype == "Apartment":
            land = 0.0
            st.caption("Land size is set to zero for apartments, which is how the "
                       "model was trained.")
        else:
            land = st.slider("Land size (sqm)", 0, 2000, 450, step=10)

        year = st.number_input("Year built", 1850, 2030, 1995)
        station = st.slider("Distance to nearest station (km)", 0.0, 10.0, 1.5, 0.1)

        st.markdown("**Features**")
        f1, f2 = st.columns(2)
        pool = int(f1.checkbox("Pool"))
        aircon = int(f1.checkbox("Air conditioning", value=True))
        reno = int(f2.checkbox("Renovated"))
        view = int(f2.checkbox("Water view"))

        with st.expander("Sale context and listing text"):
            method = st.selectbox("Sale method", META["sale_methods"])
            dom = st.number_input("Days on market", 0, 500, 30)
            sale_date = st.date_input("Sale date", pd.Timestamp("2025-08-01"))
            blurb = st.text_area(
                "Agent description",
                placeholder="Paste the listing blurb here. The model uses the "
                            "wording as a weak extra signal.",
                height=90)

        go = st.button("Estimate price", type="primary", use_container_width=True)

    with right:
        if go:
            payload = {
                "suburb": suburb, "property_type": ptype,
                "bedrooms": int(beds), "bathrooms": int(baths),
                "car_spaces": int(cars), "land_size_sqm": float(land),
                "internal_area_sqm": float(internal), "year_built": int(year),
                "has_pool": pool, "has_aircon": aircon,
                "renovated": reno, "water_view": view,
                "distance_to_station_km": float(station),
                "sale_method": method, "days_on_market": int(dom),
                "sale_date": str(sale_date), "agent_description": blurb,
            }
            with st.spinner("Estimating"):
                res = call_predict(payload, MODE)
            if res:
                render_result(res, suburb)
        else:
            st.info("Fill in the details on the left and press Estimate price.")
            st.markdown(
                "**What this tool does.** It predicts a sale price from a Ridge "
                "regression trained on 120 sales. It also tells you when it is "
                "likely to be wrong, which matters more than the number itself.\n\n"
                "**Known weak spots**, from the error analysis in the notebook:\n\n"
                "- Small dwellings in expensive suburbs, where it overshoots badly\n"
                "- Townhouses, its highest error category at around 20 percent\n"
                "- Anything near the top or bottom of the price range\n"
                "- Any suburb other than the three it was trained on")


# ---------------------------------------------------------------------------
# tab 2, batch upload
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Score a spreadsheet of properties")
    st.write("Upload a CSV with one property per row. Missing optional columns are "
             "filled with sensible defaults.")

    required = ["suburb", "property_type", "bedrooms", "bathrooms", "internal_area_sqm"]
    st.code(", ".join(required), language=None)

    template = pd.DataFrame([
        {"suburb": "Mosman", "property_type": "House", "bedrooms": 4, "bathrooms": 3,
         "car_spaces": 2, "land_size_sqm": 620, "internal_area_sqm": 260,
         "year_built": 1990, "has_pool": 1, "has_aircon": 1, "renovated": 1,
         "water_view": 1, "distance_to_station_km": 1.1,
         "sale_method": "Auction", "days_on_market": 28,
         "sale_date": "2025-08-01", "agent_description": "Renovated family home"},
        {"suburb": "Blacktown", "property_type": "House", "bedrooms": 3, "bathrooms": 1,
         "car_spaces": 1, "land_size_sqm": 520, "internal_area_sqm": 150,
         "year_built": 1972, "has_pool": 0, "has_aircon": 1, "renovated": 0,
         "water_view": 0, "distance_to_station_km": 2.4,
         "sale_method": "Private Treaty", "days_on_market": 52,
         "sale_date": "2025-07-15", "agent_description": "Original condition"},
    ])
    st.download_button("Download a template CSV",
                       template.to_csv(index=False).encode(),
                       "property_template.csv", "text/csv")

    up = st.file_uploader("Choose a CSV", type="csv")
    if up:
        raw = pd.read_csv(up)
        st.write(f"Loaded {len(raw)} rows.")

        missing = [c for c in required if c not in raw.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            defaults = {"car_spaces": 1, "land_size_sqm": 0.0, "year_built": 1995,
                        "has_pool": 0, "has_aircon": 1, "renovated": 0, "water_view": 0,
                        "distance_to_station_km": 1.5, "sale_method": "Auction",
                        "days_on_market": 30, "sale_date": "2025-08-01",
                        "agent_description": ""}
            for col, val in defaults.items():
                if col not in raw.columns:
                    raw[col] = val
            raw = raw.fillna(defaults)

            rows, failures = [], []
            bar = st.progress(0.0, "Scoring")
            for i, rec in enumerate(raw.to_dict("records")):
                payload = {k: rec[k] for k in
                           list(defaults) + required if k in rec}
                payload["sale_date"] = str(payload["sale_date"])[:10]
                payload["agent_description"] = str(payload.get("agent_description") or "")
                try:
                    res = call_predict(payload, MODE)
                    rows.append({**rec,
                                 "predicted_price": res["predicted_price"],
                                 "lower_80": res["lower_80"],
                                 "upper_80": res["upper_80"],
                                 "confidence": res["confidence"],
                                 "n_warnings": len(
                                     [w for w in res["warnings"]
                                      if not w.startswith("No specific")])})
                except Exception as exc:
                    failures.append((i + 1, str(exc)[:120]))
                bar.progress((i + 1) / len(raw), f"Scoring {i + 1} of {len(raw)}")
            bar.empty()

            if failures:
                st.warning(f"{len(failures)} rows could not be scored.")
                for ln, msg in failures[:5]:
                    st.caption(f"row {ln}: {msg}")

            if rows:
                out = pd.DataFrame(rows)
                st.success(f"Scored {len(out)} properties.")

                m1, m2, m3 = st.columns(3)
                m1.metric("Total estimated value", f"${out['predicted_price'].sum():,.0f}")
                m2.metric("Median estimate", f"${out['predicted_price'].median():,.0f}")
                m3.metric("Rows with risk flags", int((out["n_warnings"] > 0).sum()))

                show = ["suburb", "property_type", "bedrooms", "predicted_price",
                        "lower_80", "upper_80", "confidence", "n_warnings"]
                st.dataframe(
                    out[show].style.format(
                        {c: "${:,.0f}" for c in
                         ["predicted_price", "lower_80", "upper_80"]}),
                    use_container_width=True, height=380)

                st.download_button("Download results",
                                   out.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv",
                                   type="primary")


# ---------------------------------------------------------------------------
# tab 3, model card
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("About the model")

    st.markdown(f"""
The deployed model is **{META['model_name']}** on a log price target, trained on
{META['trained_on_rows']} sales collected across {', '.join(META['suburbs'])}.

I compared three approaches with 5-fold cross validation. Ridge won, which is not what
I predicted before running it. The tree models fell apart on expensive properties
because a tree cannot predict outside the range it was trained on, and Ridge can.
""")

    metrics = pd.DataFrame(META["all_model_metrics"]).T
    metrics = metrics[["MAE", "RMSE", "R2", "MAPE"]].sort_values("MAE")
    metrics.index.name = "model"

    st.markdown("##### Cross validated performance, errors in dollars")

    def highlight_winner(row):
        """Mark the best model. This is done with plain CSS on purpose.

        pandas Styler.background_gradient() pulls its colormaps from matplotlib,
        which is not in the deployment requirements because nothing else in the
        app needs it. Calling it here raised "background_gradient requires
        matplotlib" on Streamlit Cloud. Styler.apply() with literal CSS strings
        has no such dependency, and highlighting the winning row says more than a
        gradient did anyway.
        """
        best = row.name == metrics.index[0]          # already sorted by MAE
        css = f"background-color: {BLUE}1A; font-weight: 600" if best else ""
        return [css] * len(row)

    st.dataframe(
        metrics.style.format(
            {"MAE": "${:,.0f}", "RMSE": "${:,.0f}", "R2": "{:.3f}", "MAPE": "{:.1f}%"}
        ).apply(highlight_winner, axis=1),
        use_container_width=True)
    st.caption(f"{metrics.index[0]} is the deployed model, highlighted above.")
    st.caption("MAE is the typical miss in dollars. MAPE is the same thing as a "
               "percentage, which is the fairer comparison across a dataset "
               r"spanning \$400k to \$7.9m.")

    rng = META["training_price_range"]
    st.markdown("##### Where the model is reliable")
    st.markdown(f"""
| | |
|---|---|
| Training price range | \\${rng['min']:,.0f} to \\${rng['max']:,.0f} |
| Dense range (5th to 95th percentile) | \\${rng['p05']:,.0f} to \\${rng['p95']:,.0f} |
| Typical error | {META['metrics']['MAPE']:.1f}% |

Outside the dense range there are few comparable sales and the estimate should be
treated as a rough indication.
""")

    st.markdown("##### Known limitations")
    st.markdown("""
- **Three suburbs only.** Mosman, Parramatta and Blacktown. It will return a number
  for anything you give it, but it only knows these markets.
- **It does not know where in the suburb the property is.** All of Mosman is one
  place to this model. In reality a harbourside street and a main road are very
  different, and this is the single biggest missing feature.
- **Condition is a yes or no flag.** A cosmetic refresh and a full rebuild both count
  as renovated, which caused one of the five worst errors in testing.
- **Trained on sold properties only.** Listings that failed to sell are invisible to
  it, so it will over-estimate properties that would struggle to find a buyer.
- **The training data was generated, not scraped.** See the notebook for the full
  explanation. Real accuracy would be lower than the numbers above.
""")

    st.markdown("##### Ethics")
    st.markdown("""
Features like school catchment rating and distance to the CBD correlate with the
income and demographic makeup of an area. A model that learns which postcodes sell
for less is partly learning a socioeconomic pattern, and using it to set prices
rather than describe them can entrench that pattern.

The error rate is also not evenly spread. The model is least accurate at the cheapest
and most expensive ends of the range, and of those two groups the one least able to
absorb a bad valuation is first home buyers.

This is why the tool shows a range and its own failure modes instead of a single
confident number. It is built to support a human decision, not to replace one.
""")
