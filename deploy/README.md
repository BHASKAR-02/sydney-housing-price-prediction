# Deployment notes

Two pieces to host: the FastAPI backend and the Streamlit front end. Both have free
options.

## The short version

The fastest path is **Streamlit Community Cloud only**. The app detects that no API is
reachable and loads the model in process, so you get a working public URL from one
deploy and nothing to pay for. Use this if you just need a link that works.

Deploy the backend as well if you want to show the full two service architecture, which
is what the report describes.

## Free tier limits worth knowing

| Platform | What you get free | The catch |
|---|---|---|
| Streamlit Community Cloud | Unlimited public apps, 1 GB RAM | Sleeps after 7 days idle, wakes on visit |
| Render web service | 750 hours a month | Sleeps after 15 min idle, ~50s cold start |
| Hugging Face Spaces | 2 vCPU, 16 GB RAM | Public only on the free tier |
| Railway | $5 credit a month | Runs out if the service never sleeps |

The Render cold start is the reason the Streamlit app has a standalone fallback. Without
it, a marker opening the link during a cold start would see an error rather than a slow
load.

## If the build hangs on "Processing dependencies"

This one cost me time, so it is worth writing down. Streamlit Community Cloud picks
its own Python version, and it is often a very recent one. If `requirements.txt`
pins exact older versions, there is no matching wheel for that Python, so the
installer falls back to compiling numpy and scikit-learn from source. On a free tier
with 1 GB of RAM that either takes forever or runs out of memory, and the log just
sits on `Processing dependencies` with no error.

Two defences are in place:

1. **The requirements use ranges, not exact pins**, so the installer can pick a
   build that actually exists for whatever Python it has.
2. **Set the Python version explicitly.** In the Streamlit Cloud app settings, open
   *Advanced settings* and choose **Python 3.12**. This is the reliable fix. Reboot
   the app afterwards.

If you need the exact versions the report's numbers came from, use
`requirements-lock.txt` on Python 3.9 to 3.12.

## Dependency files

- **`requirements.txt`** at the repo root is what Streamlit Community Cloud installs.
  Lean and version-ranged: the app and API only.
- **`requirements-lock.txt`** exact versions for reproducing the reported results.
- **`requirements-dev.txt`** adds the notebook dependencies. Local use only.
- **`backend/requirements.txt`** is what `render.yaml` installs for the API service.

## Before you redeploy, run the smoke test

`scripts/smoke_test.py` exercises every code path the deployed app uses, in an
environment that has only the runtime requirements. Both deployment bugs I hit
would have been caught by it locally instead of on Streamlit Cloud.

```bash
python -m venv /tmp/leanenv
/tmp/leanenv/bin/pip install -r requirements.txt
/tmp/leanenv/bin/python scripts/smoke_test.py
/tmp/leanenv/bin/python scripts/smoke_test.py --break-model   # simulates a version mismatch
```

The trap worth knowing: **do not use pandas Styler methods that depend on
matplotlib** (`background_gradient`, `bar`, `text_gradient`) anywhere in the app.
matplotlib is deliberately not a runtime dependency, and those methods fail at
render time with `background_gradient requires matplotlib` rather than at import,
so nothing catches them until a user opens that tab. `Styler.apply` with literal
CSS strings is the safe equivalent.

## Why the model can survive a version mismatch

`models/best_model.joblib` is a pickle, so it is tied to the scikit-learn version
that wrote it. If the host installs a different version the unpickle can fail.
`backend/main.py` catches that and rebuilds the Ridge pipeline from
`data/sydney_housing_raw.csv` instead, which takes well under a second at 120 rows
and produces identical predictions. `/health` reports which path was used via
`model_source`.

## Files here

- `render.yaml` blueprint for the Render backend
- `Procfile` for anything that reads one (Railway, Heroku style platforms)
- `Dockerfile` only if your host wants a container

Full step by step instructions are in the main `README.md` at the repo root.
