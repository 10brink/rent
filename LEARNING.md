# Learning Notes - Rental Price Predictor (Ypsilanti)

This is a short, practical review of how the project works and why certain choices were made.
It is written for quick reference rather than deep theory.

---

## 1) Big Picture

Goal: predict a "fair rent" for a listing in Ypsilanti using recent rental data.

Pipeline flow:
1) Fetch listings from RentCast.
2) Clean/normalize data and impute missing values.
3) Cluster listings into "neighborhoods" using K-Means on lat/lon.
4) Train a CatBoost model to predict rent.
5) Save model + metadata, serve predictions via Flask API.

Key files:
- `pipeline.py`: end-to-end training and evaluation.
- `rental_price_model.py`: training utilities + prediction helpers.
- `app.py`: API server and web UI.
- `templates/index.html`: web UI.

---

## 2) Data Fetching and Fields

We fetch listings from RentCast and store:
- Address fields (line1, line2/unit, city, state, zip, formatted address)
- Location (lat, lon)
- Features (beds, baths, sqft, property type)
- Dates (listed/created/last seen/removed, days on market)

Why include dates?
- Real estate is seasonal. We can validate on *recent* listings to measure drift.

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

Target transform:
- We train on `log1p(price)` and then invert with `expm1`.
- This reduces skew and improves stability.

---

## 6) Avoiding Leakage in Evaluation

Best practice: *every learned step must be fit only on the training fold*.

We built a leakage-safe CV pipeline:
- `KMeansClusterer` fits only on train fold.
- `ImputerByCity` computes medians only on train fold.
- Outlier removal is skipped during CV (train-only if ever used).

We also added a **time-based split** using `Listed Date`:
- Train on oldest 80%, validate on newest 20%.
- This is a realistic test for a drifting market.

---

## 7) Outlier Policy (Not Interested in Luxury)

The project removes price outliers using IQR.

Note:
- This is a design choice.
- It means the model is optimized for typical rents, not luxury outliers.

---

## 8) Dedupe Strategy (Avoid Dropping Multi-Unit Listings)

We now dedupe by:
1) Listing ID (preferred)
2) Full address (line1 + line2 + city + state + zip)

Why?
- Address-only dedupe can delete valid multi-unit data.

---

## 9) Serving Predictions

`app.py` uses:
- Geocoding to find the nearest cluster
- Model + metadata to predict price and range
- Comparable listings for context

It returns:
- predicted price
- price range based on CV RMSE
- neighborhood + coordinates

---

## 10) Common Pitfalls

- Missing env var: `RENTCAST_API_KEY` must be set.
- Forgetting to retrain after changes.
- Using "N/A" strings in numeric fields.
- Leakage in evaluation (fixed by the pipeline).

---

## 11) How to Retrain

Full pipeline:
```
python3 pipeline.py --city ypsilanti
```

---

## 12) Glossary

- **Leakage**: when training uses information from the validation set.
- **CV RMSE**: average error across CV folds (lower is better).
- **Log transform**: modeling `log(price)` instead of raw price.
- **Cluster ID**: a neighborhood-like label from K-Means.

---

If you want, I can expand any section with examples or diagrams.
