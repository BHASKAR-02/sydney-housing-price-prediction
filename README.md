# Sydney Housing Price Prediction and Decision Support System

SIT720 Machine Learning, task 8.1D (Distinction).

A housing price prediction tool for three Sydney suburbs, covering the full machine
learning lifecycle: data collection, exploration, feature engineering, model comparison,
error analysis, a comparison against an LLM and against my own estimates, and a deployed
web application.

## What it does

Give it a property in Mosman, Parramatta or Blacktown and it predicts a sale price, an
80 percent confidence range, and a list of reasons the prediction might be wrong. The
warnings are the part I care about most, because the error analysis showed the model
fails in specific and predictable ways rather than randomly.

## Results

Three models compared with 5-fold cross validation, all trained on log price and
scored back in dollars:

| Model | MAE | RMSE | R squared | MAPE | Train to test gap |
|---|---|---|---|---|---|
| **Ridge Regression** | **$281,822** | **$456,404** | **0.920** | **15.9%** | **0.063** |
| Random Forest | $351,996 | $655,144 | 0.836 | 18.9% | 0.123 |
| Gradient Boosting | $352,600 | $698,258 | 0.813 | 17.1% | 0.171 |

Ridge won, which is not what I predicted. I expected Random Forest to win because of the
suburb by property type interaction. What I missed is that with 120 rows spanning $405k
to $7.9m, most of the error budget goes on properties near the edge of the training
range, and trees structurally cannot predict outside the range they saw. Gradient
Boosting reached a training R squared of exactly 1.000, memorising every row, and came
last on held out data.

Full reasoning is in the notebook, Part 3.6.

## Repository layout

```
├── data/
│   ├── sydney_housing_raw.csv        120 sold properties, 23 columns
│   └── data_dictionary.md            every column, and the known problems with them
├── src/
│   └── build_dataset.py              builds the dataset, see the honesty note below
├── notebooks/
│   └── sydney_housing_analysis.ipynb the analysis, Parts 1 to 6, runs top to bottom
├── backend/
│   ├── main.py                       FastAPI service
│   ├── features.py                   feature engineering, shared with the notebook
│   └── requirements.txt
├── frontend/
│   ├── app.py                        Streamlit application
│   └── requirements.txt
├── models/
│   ├── best_model.joblib             the fitted Ridge pipeline
│   └── model_metadata.json           metrics, schema, per segment uncertainty
├── outputs/
│   ├── fig01 to fig13 .png           figures used in the report
│   └── screenshots/                  application screenshots
├── deploy/                           Render blueprint, Procfile, Dockerfile
└── report/
    └── report.md                     the submitted report
```

## A note on the data

I could not scrape realestate.com.au or domain.com.au. Both block automated collection
in their terms of use and their sold listing pages render client side. `src/build_dataset.py`
generates the dataset instead, from a hedonic pricing structure calibrated against
published median sale prices per suburb and property type for 2023 to 2025, with feature
distributions matched to what I saw browsing sold listings by hand.

This is a real limitation and I discuss it in the notebook rather than hiding it. The
models face an easier problem than they would on genuine sales, so the accuracy numbers
above are an upper bound.

To swap in your own manually collected data, save it as `data/sydney_housing_raw.csv`
using the column names in `data/data_dictionary.md` and re-run the notebook. Nothing
else needs to change.

## Running it locally

Python 3.9 or newer.

```bash
git clone https://github.com/<your-username>/sydney-housing-price-prediction.git
cd sydney-housing-price-prediction

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Build the dataset (optional, it is already committed)

```bash
python src/build_dataset.py
```

### 2. Run the analysis

```bash
jupyter lab notebooks/sydney_housing_analysis.ipynb
```

Run all cells. Takes about 30 seconds. It writes the figures to `outputs/` and the
trained model to `models/`.

### 3. Start the API

```bash
uvicorn backend.main:app --reload --port 8000
```

Interactive docs at http://localhost:8000/docs

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"suburb":"Mosman","property_type":"House","bedrooms":4,"bathrooms":3,
       "car_spaces":2,"land_size_sqm":600,"internal_area_sqm":250,
       "year_built":1995,"water_view":1,"renovated":1}'
```

### 4. Start the app

In a second terminal:

```bash
streamlit run frontend/app.py
```

Opens on http://localhost:8501. It finds the API automatically. If the API is not
running it loads the model in process instead, so the app works either way.

## Using the application

**Estimate one property.** Fill in the form on the left, press Estimate price. You get
the predicted price, an 80 percent range, and any risk flags that apply. The range is
calculated per suburb and property type rather than globally, because testing showed
error varies a lot between segments.

**Upload a CSV.** Score a whole spreadsheet at once. Download the template first to get
the column names right. Only `suburb`, `property_type`, `bedrooms`, `bathrooms` and
`internal_area_sqm` are required, everything else falls back to a default.

**About the model.** Model card with the cross validation results, the reliable price
range, known limitations and the ethical considerations.

## API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and whether the model loaded |
| GET | `/metadata` | Suburbs, property types, metrics, defaults |
| POST | `/predict` | One property |
| POST | `/predict/batch` | Up to 500 properties |
| GET | `/docs` | Swagger UI |

## Deploying it

See `deploy/README.md` for the detail. Two options:

**Fastest.** Deploy the Streamlit app to Streamlit Community Cloud on its own. The app
runs the model in process when no API is reachable, so one deploy gets you a working
public URL for free.

**Full architecture.** Backend to Render using `deploy/render.yaml`, then the Streamlit
app with `API_URL` pointed at the Render URL. This is what the report describes.

## Known limitations

- **Three suburbs only.** It will return a number for anything, but it only knows these
  three markets.
- **No position within a suburb.** All of Mosman is one place to this model. This is the
  biggest missing feature and the notebook explains why.
- **Condition is a binary flag.** A cosmetic refresh and a full rebuild both count as
  renovated. This caused one of the five worst errors.
- **Small sample.** 120 rows. Some feature combinations appear once or twice.
- **Sold properties only.** Withdrawn listings are invisible, so it over-estimates
  properties that would struggle to sell.

## GenAI acknowledgement

I used Claude (Anthropic) for planning the notebook structure, suggesting engineered
features to try, debugging a scikit-learn ColumnTransformer error, and proofreading. I
also used it as the LLM valuation system in Part 5, which is what that comparison is
about.

All modelling decisions, the suburb selection, the feature engineering rationale, the
interpretation of results and the conclusions are mine. I checked every suggestion
against the actual output, and rejected several, including an early recommendation to
drop the outlier properties, which I argue against in Part 2 of the notebook.
