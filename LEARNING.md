# Learning Notes - Rental Price Predictor (Michigan)

This is a short, practical review of how the project works and why certain
choices were made. It is written for quick reference rather than deep theory.

---

## 1) Big Picture

Goal: predict a "fair rent" for a listing in supported Michigan cities.

Pipeline flow (per city):
1) Fetch listings from RentCast (timestamped under `data/<city>/raw/`).
2) Clean/normalize data and impute missing values.
3) Cluster listings into "neighborhoods" using K-Means on lat/lon.
4) Train a CatBoost model to predict rent.
5) Save model + metadata, serve predictions via Flask API.

Optional:
- Train a pooled Single Family model for Lansing + East Lansing.
- Use comps-average predictions in the API.

Key files:
- `pipeline.py`: end-to-end training and evaluation.
- `rental_price_model.py`: training utilities + prediction helpers.
- `app.py`: API server and web UI.
- `city_config.py` + `config/cities.json`: multi-city config.
- `evaluate_comps_knn.py`: comps-average evaluation utilities.
- `train_pooled_sfh.py`: pooled Single Family model training.
- `templates/index.html`: web UI.

---

## 2) Data Fetching and Fields

We fetch listings from RentCast and store:
- Address fields (line1, line2/unit, city, state, zip, formatted address)
- Location (lat, lon)
- Features (beds, baths, sqft, property type)
- Dates (listed/created/last seen/removed, days on market)

Why include dates?
- Real estate is seasonal. We can validate on recent listings to measure drift.

Processed data lives at `data/<city>/rentals_with_stats.csv`.

---

## 3) Cleaning + Imputation (Missing Data)

Common issue: missing or non-numeric values (e.g., "N/A").

We do:
- Replace "N/A" with real NaN.
- Coerce numeric fields to numeric.
- Impute missing beds/baths/sqft using city-specific medians.

Why city medians?
- They reflect local housing stock better than a generic default.

---

## 4) Neighborhood Clustering

Raw lat/lon is too granular. We cluster listings with K-Means so the model can
learn neighborhood-level price effects.

We:
- Fit K-Means on lat/lon.
- Assign `Cluster_ID`.
- Reverse geocode cluster centroids for human-readable names.

Why?
- More robust than raw coordinates with a small dataset.

---

## 5) Modeling

We use CatBoost because it:
- Handles categorical features well (property type, cluster ID).
- Performs strongly without heavy feature engineering.

Core features (per-city models):
- `Beds`, `Baths`, `SqFt`, `Property Type`, `Cluster_ID`

Pooled SFH model adds:
- `City_Slug` (categorical, to share signal across cities)

Target transform:
- We train on `log1p(price)` and then invert with `expm1`.
- This reduces skew and improves stability.

---

## 6) Avoiding Leakage in Evaluation

Best practice: every learned step must be fit only on the training fold.

We built a leakage-safe CV pipeline:
- `KMeansClusterer` fits only on train fold.
- `ImputerByCity` computes medians only on train fold.
- Outlier removal is skipped during CV (train-only if ever used).

We also added a time-based split using `Listed Date`:
- Train on oldest 80%, validate on newest 20% (when enough data exists).
- This is a realistic test for a drifting market.

---

## 7) Outlier Policy (Not Interested in Luxury)

The project removes price outliers using IQR.

Note:
- This is a design choice.
- It means the model is optimized for typical rents, not luxury outliers.

---

## 8) Dedupe Strategy (Avoid Dropping Multi-Unit Listings)

We dedupe by:
1) Listing ID (preferred)
2) Full address (line1 + line2 + city + state + zip)

Why?
- Address-only dedupe can delete valid multi-unit data.

---

## 9) Serving Predictions

`app.py` does:
- Geocode the address (Nominatim).
- Detect the nearest supported city within its radius.
- Find the nearest cluster.
- Compute two estimates:
  - Model prediction (CatBoost + CV RMSE range)
  - Comps average (KNN by beds/baths/sqft + distance; cluster preferred)

Prediction modes:
- `auto` (default): uses comps for East Lansing single-family rentals when
  available; otherwise uses the model. For Lansing/East Lansing SFH, it will
  use the pooled SFH model if available.
- `model`: always use the model.
- `comps`: always use comps (errors if insufficient comps).

Other endpoints:
- `GET /api/cities` lists configured cities and whether a trained model exists.
- `/methodology` shows per-city CV RMSE and comps metrics.

---

## 10) Common Pitfalls

- Missing env var: `RENTCAST_API_KEY` must be set in `.env.local`.
- Address outside configured city radii (API returns an error).
- No trained model for the target city (missing `models/<city>/`).
- `comps` mode with too few comparable listings.
- Forgetting to retrain after changing features or preprocessing.

---

## 11) Feature Candidates in the CSV (Ann Arbor)

Snapshot: `data/ann_arbor/raw/rentals_raw_20260131.csv` (2026-01-31)

Good candidates (high coverage, likely signal):
- `Days On Market` (0% missing): strong pricing/recency proxy.
- `Listed Date` / `Created Date` / `Last Seen Date` (0% missing): derive recency + seasonality.
- `Zip` (0% missing): strong location proxy (works well as categorical).
- `Address 2` (26.6% missing): can derive `has_unit` or unit count flag.

Medium candidates (sparse, optional):
- `Year Built` (~83% missing): only worth it with bucketing + imputation.

Not usable right now (no data):
- `Amenities` (100% missing)
- `Days Old` (100% missing)
- `Removed Date` (100% missing)

---

## 12) How to Retrain

Full pipeline:
```
python3 pipeline.py --city ypsilanti
```

Other cities:
```
python3 pipeline.py --city ann_arbor
python3 pipeline.py --city detroit
python3 pipeline.py --city lansing
python3 pipeline.py --city east_lansing
```

Optional pooled SFH model:
```
python3 train_pooled_sfh.py
```

---

## 13) Glossary

- **Leakage**: when training uses information from the validation set.
- **CV RMSE**: average error across CV folds (lower is better).
- **Log transform**: modeling `log(price)` instead of raw price.
- **Cluster ID**: a neighborhood-like label from K-Means.
- **Comps average**: mean price of nearby comparable listings.
- **Auto mode**: API chooses between model and comps based on rules.

---

If you want, I can expand any section with examples or diagrams.
