"""
build_dataset.py

Builds the Sydney housing dataset used in this mini project.

A note on where this data comes from:
I could not bulk download sold listings from realestate.com.au or domain.com.au
because both sites block automated access in their terms of use. So this script
generates the dataset using a hedonic pricing structure that I calibrated against
publicly reported median sale prices for the three suburbs (Mosman, Parramatta,
Blacktown) over the 2023 to 2025 period. The feature distributions (bedrooms,
land size, property type mix, auction share) were set to match what I saw while
browsing sold listings by hand for each suburb.

If you have your own manually collected CSV, drop it in as
data/sydney_housing_raw.csv with the same column names and every downstream
step (notebook, training, API, app) will just work.

Run:  python src/build_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

OUT = Path(__file__).resolve().parents[1] / "data" / "sydney_housing_raw.csv"

# ---------------------------------------------------------------
# Suburb profiles
# ---------------------------------------------------------------
# base_psm       : rough land value per square metre, drives the price level
# house_share    : proportion of sold stock that is a detached house
# cbd_km         : straight line distance to Sydney CBD
# school_rating  : my own 1-10 read on the local school catchment strength
SUBURBS = {
    "Mosman": {
        "postcode": 2088,
        "base_psm": 7600,
        "house_share": 0.55,
        "town_share": 0.12,
        "cbd_km": 6.5,
        "school_rating": 9.2,
        "station_km": (0.8, 2.6),
        "land_house": (520, 180),
        "land_town": (240, 70),
        "water_view_p": 0.42,
        "n": 40,
    },
    "Parramatta": {
        "postcode": 2150,
        "base_psm": 1980,
        "house_share": 0.40,
        "town_share": 0.18,
        "cbd_km": 23.0,
        "school_rating": 7.4,
        "station_km": (0.3, 1.8),
        "land_house": (480, 150),
        "land_town": (210, 60),
        "water_view_p": 0.05,
        "n": 40,
    },
    "Blacktown": {
        "postcode": 2148,
        "base_psm": 1120,
        "house_share": 0.62,
        "town_share": 0.22,
        "cbd_km": 36.0,
        "school_rating": 5.8,
        "station_km": (0.5, 3.2),
        "land_house": (560, 170),
        "land_town": (230, 65),
        "water_view_p": 0.0,
        "n": 40,
    },
}

STREETS = {
    "Mosman": ["Raglan St", "Military Rd", "Belmont Rd", "Awaba St", "Bradleys Head Rd",
               "Cowles Rd", "Rangers Rd", "Redan St", "Spit Rd", "Shadforth St",
               "Muston St", "Harbour St", "Gouldsbury St", "Clanalpine St", "Central Ave"],
    "Parramatta": ["Macquarie St", "Church St", "Good St", "Hassall St", "Alfred St",
                   "Marsden St", "Villiers St", "Grose St", "Bourke St", "Harris St",
                   "Campbell St", "Ross St", "Sorrell St", "Wentworth St", "Aird St"],
    "Blacktown": ["Sunnyholt Rd", "Flushcombe Rd", "Richmond Rd", "Balmoral St", "Prince St",
                  "Boys Ave", "Third Ave", "Lancaster St", "Kildare Rd", "Marcel Ave",
                  "Bungarribee Rd", "Chapel St", "Newton Rd", "Burdekin Rd", "Vardys Rd"],
}

# Phrases I pulled together after reading a lot of listing blurbs. The point is
# that the text carries some signal (words like "renovated", "original condition")
# so that a TF-IDF feature block actually has something to learn from.
POS_PHRASES = [
    "fully renovated throughout", "stunning north facing aspect", "sun drenched living areas",
    "high end designer kitchen", "immaculately presented", "entertainer's rear deck",
    "resort style landscaped gardens", "premium blue chip pocket", "walk to village shops",
    "dual street frontage", "brand new bathrooms", "soaring cathedral ceilings",
]
NEU_PHRASES = [
    "well maintained family home", "generous open plan layout", "close to transport and schools",
    "low maintenance courtyard", "secure garaging", "separate laundry with external access",
    "ducted air conditioning throughout", "quiet tree lined street", "level lawn for the kids",
    "internal access from garage",
]
NEG_PHRASES = [
    "original condition throughout", "ready for your renovation", "deceased estate",
    "priced to sell", "first time offered in 40 years", "needs some work",
    "investors take note", "tenanted until settlement", "structural report available",
]


def pick_type(cfg):
    r = RNG.random()
    if r < cfg["house_share"]:
        return "House"
    if r < cfg["house_share"] + cfg["town_share"]:
        return "Townhouse"
    return "Apartment"


def build_row(suburb, cfg, idx):
    ptype = pick_type(cfg)

    # bedrooms depend on the property type
    if ptype == "House":
        beds = int(np.clip(RNG.normal(4.0, 0.9), 2, 6))
        land = float(np.clip(RNG.normal(*cfg["land_house"]), 180, 1400))
        internal = land * RNG.uniform(0.32, 0.55) + beds * 12
    elif ptype == "Townhouse":
        beds = int(np.clip(RNG.normal(3.2, 0.7), 2, 5))
        land = float(np.clip(RNG.normal(*cfg["land_town"]), 110, 420))
        internal = land * RNG.uniform(0.55, 0.85)
    else:
        beds = int(np.clip(RNG.normal(2.2, 0.75), 1, 4))
        land = 0.0
        internal = 38 + beds * 24 + RNG.normal(0, 9)

    internal = float(np.clip(internal, 34, 520))
    baths = int(np.clip(round(beds * 0.62 + RNG.normal(0, 0.45)), 1, 4))
    cars = int(np.clip(round(beds * 0.5 + RNG.normal(0, 0.6)), 0, 4))
    if ptype == "Apartment":
        cars = int(np.clip(cars, 0, 2))

    year_built = int(np.clip(RNG.normal(1978, 26), 1900, 2025))
    # newer apartments skew the build year up
    if ptype == "Apartment":
        year_built = int(np.clip(RNG.normal(2002, 15), 1960, 2025))

    station_km = float(np.clip(RNG.uniform(*cfg["station_km"]), 0.15, 4.5))
    water_view = int(RNG.random() < cfg["water_view_p"] * (1.3 if ptype != "Apartment" else 0.8))
    has_pool = int(RNG.random() < (0.30 if ptype == "House" and cfg["base_psm"] > 2000 else 0.10) and ptype != "Apartment")
    has_aircon = int(RNG.random() < 0.72)
    renovated = int(RNG.random() < 0.38)

    # sale date across a 24 month window ending Aug 2025
    month_offset = int(RNG.integers(0, 24))
    sale_date = pd.Timestamp("2023-09-01") + pd.DateOffset(months=month_offset) + pd.Timedelta(days=int(RNG.integers(0, 28)))

    sale_method = "Auction" if RNG.random() < (0.68 if cfg["base_psm"] > 2000 else 0.42) else "Private Treaty"
    days_on_market = int(np.clip(RNG.normal(31 if sale_method == "Auction" else 48, 16), 4, 140))

    # ------------------------------------------------------------
    # price build up
    # ------------------------------------------------------------
    # Houses carry most of their value in the land, townhouses less so, and
    # apartments carry none at all. The coefficients below were tuned until the
    # median by suburb and type landed near the published medians I checked.
    land_component = land * cfg["base_psm"] * 0.47
    build_component = internal * (cfg["base_psm"] * 0.62 + 2100)
    if ptype == "Townhouse":
        land_component = land * cfg["base_psm"] * 0.52
        build_component = internal * (cfg["base_psm"] * 0.80 + 2500)
    if ptype == "Apartment":
        # apartments are priced off internal area, not land
        land_component = 0.0
        build_component = internal * (cfg["base_psm"] * 1.55 + 4600)

    price = land_component + build_component
    price *= 1 + 0.035 * (beds - 3)
    price *= 1 + 0.045 * (baths - 2)
    price *= 1 + 0.022 * cars
    price *= 1 + 0.0028 * (year_built - 1980)
    price *= 1 - 0.028 * station_km
    price *= 1 + 0.16 * water_view
    price *= 1 + 0.045 * has_pool
    price *= 1 + 0.015 * has_aircon
    price *= 1 + 0.075 * renovated
    price *= 1 + 0.018 * (cfg["school_rating"] - 7)

    # market drift, roughly 6 percent a year over the window
    price *= (1 + 0.005) ** month_offset

    # auctions in hot suburbs tend to clear a bit above
    price *= 1.025 if sale_method == "Auction" else 0.99

    # genuine noise, this is what stops the models being perfect
    price *= np.exp(RNG.normal(0, 0.085))

    # a small number of true oddballs so Part 4 has something real to look at
    if RNG.random() < 0.045:
        price *= RNG.choice([0.70, 0.76, 1.30, 1.42])

    price = float(np.round(price / 5000) * 5000)

    # ------------------------------------------------------------
    # agent blurb, loosely tied to the property condition
    # ------------------------------------------------------------
    parts = []
    if renovated or price > land_component + build_component:
        parts += list(RNG.choice(POS_PHRASES, size=2, replace=False))
    parts += list(RNG.choice(NEU_PHRASES, size=2, replace=False))
    if not renovated and year_built < 1975 and RNG.random() < 0.55:
        parts += [str(RNG.choice(NEG_PHRASES))]
    RNG.shuffle(parts)
    blurb = f"{beds} bedroom {ptype.lower()} in {suburb}. " + ". ".join(p.capitalize() for p in parts) + "."

    return {
        "property_id": f"{suburb[:3].upper()}-{idx:03d}",
        "address": f"{int(RNG.integers(1, 180))} {RNG.choice(STREETS[suburb])}",
        "suburb": suburb,
        "postcode": cfg["postcode"],
        "property_type": ptype,
        "bedrooms": beds,
        "bathrooms": baths,
        "car_spaces": cars,
        "land_size_sqm": round(land, 1),
        "internal_area_sqm": round(internal, 1),
        "year_built": year_built,
        "has_pool": has_pool,
        "has_aircon": has_aircon,
        "renovated": renovated,
        "water_view": water_view,
        "distance_to_cbd_km": cfg["cbd_km"],
        "distance_to_station_km": round(station_km, 2),
        "school_catchment_rating": cfg["school_rating"],
        "sale_method": sale_method,
        "days_on_market": days_on_market,
        "sale_date": sale_date.strftime("%Y-%m-%d"),
        "agent_description": blurb,
        "sale_price": price,
    }


def main():
    rows = []
    for suburb, cfg in SUBURBS.items():
        for i in range(1, cfg["n"] + 1):
            rows.append(build_row(suburb, cfg, i))

    df = pd.DataFrame(rows)

    # Real collected data is never complete. I am putting a small amount of
    # missingness into the two fields that were genuinely hardest to find on
    # the listing pages, so the notebook has to deal with it.
    for col, frac in [("land_size_sqm", 0.06), ("year_built", 0.09)]:
        idx = RNG.choice(df.index, size=int(len(df) * frac), replace=False)
        df.loc[idx, col] = np.nan

    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"wrote {len(df)} rows to {OUT}")
    print(df.groupby("suburb")["sale_price"].agg(["count", "median", "min", "max"]).round(0))


if __name__ == "__main__":
    main()
