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

## Files here

- `render.yaml` blueprint for the Render backend
- `Procfile` for anything that reads one (Railway, Heroku style platforms)
- `Dockerfile` only if your host wants a container

Full step by step instructions are in the main `README.md` at the repo root.
