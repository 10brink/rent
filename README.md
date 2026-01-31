# Rental Price Predictor (Ypsilanti, MI)
<img width="934" height="798" alt="image" src="https://github.com/user-attachments/assets/165cfc12-a48c-4a2a-b58d-eea3328baefc" />
<img width="934" height="738" alt="image" src="https://github.com/user-attachments/assets/e5ed89e3-c1e1-4b0a-a11f-a1dc2c4e5cc4" />

A local rent estimate for Ypsilanti. It pulls real listings from RentCast,
clusters them into neighborhoods, trains a CatBoost model, and serves predictions
via a Flask API + simple web UI.

## Highlights

- Uses real rental listings (RentCast API)
- Learns neighborhood effects via K-Means clustering on lat/lon
- Log-price modeling to reduce skew
- Leakage-safe evaluation (clustering + imputation fit inside CV folds)
- Simple Flask API and web form

## Quick Start

### 1) Install
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Set up environment
Create `.env.local` and add your RentCast key:
```
RENTCAST_API_KEY=your_key_here
```

### 3) Train the model
```
python3 pipeline.py --city ypsilanti
```

### 4) Run the API
```
python3 app.py
```
Server runs at `http://localhost:5001`.

## API Usage

### POST /api/predict
Request:
```json
{
  "address": "410 Olive St, Ypsilanti, MI",
  "beds": 2,
  "baths": 1,
  "sqft": 900,
  "property_type": "Apartment"
}
```

Response:
```json
{
  "predicted_price": 1220,
  "price_range": {"low": 981, "high": 1459},
  "city": "ypsilanti",
  "neighborhood": "Midtown",
  "coordinates": {"lat": 42.2476, "lon": -83.618},
  "comparable_properties": [...]
}
```

## How It Works

CatBoost is the gradient-boosting library we use. It handles categorical features natively (property type, cluster ID) and keeps training stable with default settings, which makes it a reliable fit for our rent estimator.

### Stats/ML Concepts

- **K-Means**: we cluster latitude/longitude pairs into neighborhood buckets so the model learns area-level trends instead of exact coordinates.
- **Skew**: rent values are right-skewed (there are more low rents than very expensive ones). Modeling `log(price)` balances that distribution and gives CatBoost smoother gradients.
- **Leakage**: happens when validation data sneaks into training (e.g., imputation or clustering fit on the whole dataset). We prevent it by re-fitting those transforms *inside each CV fold*.
- **CV fold**: short for cross-validation fold. We split the data into 5 folds and iteratively train on 4 folds while validating on the 5th, ensuring each row is tested once.

1) **Fetch listings** from RentCast  
2) **Cluster neighborhoods** (K-Means on lat/lon)  
3) **Clean + impute** missing data with city medians  
4) **Train CatBoost** on log(price)  
5) **Evaluate** with leakage-safe CV + time-based split  
6) **Serve predictions** via Flask

For a deeper walkthrough, see `LEARNING.md`.

## Running the Pipeline

```
python3 pipeline.py --city ypsilanti
```

Useful flags:
```
--dry-run         Fetch data only, do not train
--skip-backup     Skip model backup
--append          Append new listings to existing data
--max-pages N     Limit API pagination
```

## Project Structure

```
rent/
├── pipeline.py              # End-to-end training pipeline
├── app.py                   # Flask API server
├── rental_price_model.py    # ML utilities + prediction helpers
├── city_config.py           # Multi-city config
├── templates/               # Web UI
├── static/                  # UI assets
├── models/                  # Trained models + metadata
└── data/                    # Raw + processed data
```

## Notes and Limitations

- Optimized for **typical** rents (outliers removed).
- Currently trained for **Ypsilanti, MI** only.
- Predictions depend on RentCast coverage and data quality.

## Deploy (Render)

This repo includes `render.yaml`. The default start command is:
```
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1
```

Set `RENTCAST_API_KEY` in Render’s environment variables.

---

Questions or improvements? See `improvementideas.md` or open an issue.
