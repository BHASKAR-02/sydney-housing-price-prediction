# Data dictionary

One row per sold property. 120 rows, 40 from each of Mosman, Parramatta and Blacktown.

## Where this data comes from

Read this before using the numbers. Both realestate.com.au and domain.com.au block
automated collection in their terms of use, and their sold listing pages render client
side, so I could not pull a few hundred records cleanly. `src/build_dataset.py` generates
the dataset instead, from a hedonic pricing structure calibrated against publicly reported
median sale prices per suburb and property type over 2023 to 2025, with feature
distributions matched to what I saw browsing sold listings by hand.

This matters for how the results read. The data follows a known structure, so the models
face an easier problem than they would on real sales, and the accuracy figures in the
notebook should be treated as an upper bound.

**To use your own manually collected data instead:** save it as
`data/sydney_housing_raw.csv` with the column names below. Nothing downstream needs to
change.

## Columns

| Column | Type | Notes |
|---|---|---|
| `property_id` | string | Suburb prefix plus a sequence number, for example MOS-016 |
| `address` | string | Street number and name. No unit numbers, kept deliberately coarse |
| `suburb` | category | Mosman, Parramatta or Blacktown |
| `postcode` | int | 2088, 2150 or 2148 |
| `property_type` | category | House, Townhouse or Apartment |
| `bedrooms` | int | 1 to 6 |
| `bathrooms` | int | 1 to 4 |
| `car_spaces` | int | 0 to 4. Garage, carport and covered spaces all counted the same |
| `land_size_sqm` | float | Zero for apartments, which is a real value not a missing one. About 6 percent missing |
| `internal_area_sqm` | float | Floor area of the dwelling |
| `year_built` | int | About 9 percent missing, mostly on older stock where the listing did not say |
| `has_pool` | binary | |
| `has_aircon` | binary | |
| `renovated` | binary | Too coarse. A cosmetic refresh and a full rebuild both score 1, and this caused one of the five worst prediction errors |
| `water_view` | binary | Also too coarse. A glimpse between buildings and a full harbour frontage both score 1 |
| `distance_to_cbd_km` | float | Fixed per suburb, so it is really a second encoding of suburb |
| `distance_to_station_km` | float | Nearest train station |
| `school_catchment_rating` | float | 1 to 10, my own judgement. Also fixed per suburb |
| `sale_method` | category | Auction or Private Treaty |
| `days_on_market` | int | Listing date to sale date |
| `sale_date` | date | September 2023 to August 2025 |
| `agent_description` | text | Listing blurb. Used with TF-IDF in the notebook |
| `sale_price` | float | **Target.** AUD, rounded to the nearest 5,000 |

## Known issues

- **Three location features are the same thing.** `suburb`, `distance_to_cbd_km` and
  `school_catchment_rating` each take exactly three values, one per suburb. They are
  collinear by construction, which is part of why a penalised linear model does well here.
- **No position within a suburb.** All of Mosman is one place to this dataset. This is the
  single biggest missing feature.
- **Sold properties only.** Withdrawn listings are invisible, so anything trained on this
  will over-estimate properties that would struggle to sell.
- **The blurb repeats the suburb name.** Every description starts "N bedroom TYPE in
  SUBURB", so a TF-IDF block on this column mostly re-encodes the suburb rather than
  adding new signal. Strip the suburb and type names before vectorising if you extend this.
