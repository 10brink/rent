````markdown
# CLAUDE.md - Ypsilanti Rental Price Predictor

## For Every Nick Who Reads This

This is a rental price prediction system for Ypsilanti, Michigan. Think of it as a "Zestimate for renters" - you give it an address and property details, it tells you what the rent should be.

---

## The 30-Second Pitch

**Problem**: You're looking at a rental listing. Is $1,400/month for that 2-bedroom apartment a fair price, or are you getting fleeced?

**Solution**: This project fetches real rental data from the RentCast API, clusters properties into neighborhoods based on location, trains a CatBoost model on the data, and serves predictions via a Flask API.

**One Command to Rule Them All**:
```bash
python3 pipeline.py  # Fetches data, clusters, trains, saves - all in one go
```

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PIPELINE.PY                              │
│  (The Orchestrator - runs every 2-3 months for fresh data)      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌───────────────────┐    ┌─────────────┐  │
│   │  RentCast    │ -> │  K-Means          │ -> │  CatBoost   │  │
│   │  API Fetch   │    │  Clustering       │    │  Training   │  │
│   │  (222 listings)   │  (8 neighborhoods)│    │  (500 iters)│  │
│   └──────────────┘    └───────────────────┘    └─────────────┘  │
│         ↓                      ↓                      ↓         │
│   rentals_rentcast.csv  rentals_with_stats.csv  rental_price_   │
│                                                  model.cbm      │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                          APP.PY                                  │
│  (Flask server - the prediction API)                            │
├─────────────────────────────────────────────────────────────────┤
│   POST /api/predict                                             │
│   {address, beds, baths, sqft, property_type}                   │
│        ↓                                                        │
│   Geocode address → Find nearest cluster → Predict price        │
│        ↓                                                        │
│   {"predicted_price": 1220, "price_range": {low: 978, high: 1462}}│
└─────────────────────────────────────────────────────────────────┘
```

### The "Neighborhood" Trick

Here's the clever bit: location matters for rent, but lat/long coordinates are too granular. A model can't learn that "42.2429, -83.6199" means "downtown Ypsilanti."

**Solution**: K-Means clustering groups properties into 8 neighborhoods based on their coordinates. Each cluster gets a human-readable name via reverse geocoding (Nominatim/OpenStreetMap):

| Cluster ID | Real Name | Count |
|------------|-----------|-------|
| 0 | Midtown | 112 |
| 5 | Miles/Garland | 31 |
| 4 | Ypsilanti Charter Township | 25 |
| 7 | West Willow | 19 |
| 1 | Westlawn | 16 |
| 2 | Gault Village | 10 |
| 3 | Pittsfield Charter Township | 8 |
| 6 | Frain Lake | 1 |

When you predict for a new address, we geocode it, find the nearest cluster centroid, and use that cluster ID as a feature.

---

## Codebase Structure

```
rent/
├── pipeline.py              # The big red button - runs entire workflow
├── app.py                   # Flask API server
├── rental_price_model.py    # Model training, evaluation, prediction utilities
├── rentcast_scraper.py      # Fetches listings from RentCast API
├── cluster_neighborhoods.py # Standalone clustering script (legacy)
├── fetch_neighborhood_stats.py # Walk Score scraper (experimental)
├── zillow_scraper.py        # Zillow scraper (deprecated - they block bots)
│
├── rental_price_model.cbm   # Trained CatBoost model (~577KB)
├── rental_price_model_metadata.json  # Feature names, cluster centroids, CV RMSE
│
├── rentals_rentcast.csv     # Raw listings from last API fetch
├── rentals_with_stats.csv   # Listings with cluster assignments
│
├── backups/                 # Timestamped model backups
│   └── rental_price_model_YYYYMMDD.cbm
├── data/                    # Timestamped raw data
│   └── rentals_raw_YYYYMMDD.csv
│
├── templates/
│   └── index.html           # Web form UI
│
└── catboost_info/           # CatBoost training artifacts (can delete)
```

---

## Technologies Used

| Technology | Why |
|------------|-----|
| **CatBoost** | Handles categorical features natively (property type, cluster ID) without one-hot encoding. Beats XGBoost and Random Forest in our tests. |
| **scikit-learn** | K-Means clustering, cross-validation, train/test split |
| **Flask** | Simple API server. Overkill avoided - no need for Django here. |
| **RentCast API** | Paid API (~$50/month) that provides actual rental listings. Way more reliable than scraping. |
| **Nominatim** | Free reverse geocoding from OpenStreetMap. 1 req/sec rate limit. |
| **pandas** | Data manipulation. The usual suspect. |

---

## Decision Rationale

### Why CatBoost over XGBoost/Random Forest?

We tested all four (see `rental_price_model.py --remove-outliers`):

| Model | CV RMSE |
|-------|---------|
| CatBoost | $239 |
| XGBoost | $248 |
| Random Forest | $261 |
| Linear Regression | $293 |

CatBoost won, plus it handles categorical features without preprocessing gymnastics.

### Why cluster neighborhoods instead of using raw lat/long?

1. **Interpretability**: "Midtown" means something. "42.2429" doesn't.
2. **Generalization**: The model learns "this neighborhood is expensive" rather than overfitting to exact coordinates.
3. **Sparsity**: With only ~220 listings, raw coordinates would be too sparse.

### Why remove outliers?

The IQR method removes ~5% of listings with extreme prices (luxury homes, likely errors). Without this, the model wastes capacity learning outliers instead of the typical rental market.

### Why RentCast instead of scraping Zillow/Craigslist?

Zillow blocks bots aggressively (see `zillow_scraper.py` - it's a graveyard). RentCast costs money but provides clean, structured data with coordinates already included.

---

## Lessons Learned

### Bug #1: numpy.int64 is not JSON serializable

**Symptom**: Pipeline crashes when saving metadata.

**Cause**: `cluster_id = df['Cluster_ID'].unique()[0]` returns `numpy.int64`, not `int`. JSON doesn't know what to do with it.

**Fix**: Always cast: `int(cluster_id)` and `float(cv_rmse)`.

**Lesson**: When saving anything to JSON, wrap numeric values in native Python types.

---

### Bug #2: Geocoding failures cascade

**Symptom**: "Could not geocode address" for valid addresses.

**Cause**: Nominatim is picky. "410 Olive St, Ypsilanti, MI" fails, but "410 Olive Street, Ypsilanti, Michigan" works.

**Fix**: `normalize_address()` tries multiple variations - expanding abbreviations (St → Street), removing unit numbers, expanding state codes.

**Lesson**: When calling external APIs, build retry logic with fallback variations.

---

### Bug #3: Cluster centroids shift between training runs

**Symptom**: Model predicts $800 for an apartment that should be $1,200.

**Cause**: K-Means was initialized with `random_state=None`. Different runs produced different clusters. A new listing would map to cluster 3, but the model was trained when cluster 3 was in a different location.

**Fix**: Always `random_state=42` (or any fixed seed).

**Lesson**: Reproducibility isn't just for papers. It prevents production bugs.

---

### Pitfall: Don't trust "Year Built"

RentCast returns "N/A" for most year_built values. We tried using it as a feature but it hurt performance (the model learned to associate "N/A" with certain prices, which doesn't generalize). We dropped it.

**Lesson**: Missing data isn't just noise - it can be signal of a different distribution. Test both including and excluding sparse features.

---

### Best Practice: Backup before retrain

The pipeline automatically backs up the previous model to `backups/`. This saved us when a bad API response (only 12 listings due to a temporary glitch) produced a garbage model. We rolled back with:

```bash
cp backups/rental_price_model_20260115.cbm rental_price_model.cbm
```

---

### Best Practice: CV RMSE as confidence interval

We expose `cv_rmse` ($239) in the API response as a price range:

```json
{
  "predicted_price": 1220,
  "price_range": {"low": 981, "high": 1459}
}
```

This is more honest than a single number. It says "we're ~68% confident the true price is in this range."

---

## Running the Project

### Full pipeline (every 2-3 months)
```bash
python3 pipeline.py
```

### Start the API server
```bash
python3 app.py
# Server runs at http://localhost:5001
```

### Make a prediction
```bash
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"address": "410 Olive St, Ypsilanti, MI", "beds": 2, "baths": 1, "sqft": 900, "property_type": "Apartment"}'
```

### Retrain model only (using existing data)
```bash
python3 rental_price_model.py --remove-outliers --save
```

---

## API Reference

### POST /api/predict

**Request**:
```json
{
  "address": "410 Olive St, Ypsilanti, MI",
  "beds": 2,
  "baths": 1,
  "sqft": 900,
  "property_type": "Apartment"
}
```

**Response**:
```json
{
  "predicted_price": 1220,
  "price_range": {"low": 981, "high": 1459},
  "neighborhood_cluster": 0,
  "coordinates": {"lat": 42.2476, "lon": -83.618}
}
```

**Property Types**: `Apartment`, `Single Family`, `Condo`, `Townhouse`, `Multi Family`

---

## The Analogy

Think of this system like a real estate agent's intuition, but codified:

1. **Data Collection** (RentCast): The agent drives around, notes prices on signs
2. **Clustering** (K-Means): The agent mentally groups areas - "the college district", "suburban area", "downtown"
3. **Training** (CatBoost): The agent builds intuition - "2-bed apartments downtown go for ~$1,200"
4. **Prediction** (API): A client asks "is $1,400 fair?" The agent says "that's high for the area"

The difference: our agent never forgets, never has a bad day, and processes 220 listings in 30 seconds.

---

## Future Ideas (Not Yet Implemented)

- [ ] Add Walk Score as a feature (see `fetch_neighborhood_stats.py` - works but slow)
- [ ] Historical price trends (requires storing data over months)
- [ ] Compare to asking price and flag overpriced listings
- [ ] Expand beyond Ypsilanti (requires multi-city training)

---

*Last updated: 2026-01-30*
*Model CV RMSE: $239.26*
*Training samples: 211 (after outlier removal)*

## Deploy on Render (quick)

1. Ensure the trained model files are present in the repo: `rental_price_model.cbm` and `rental_price_model_metadata.json`.

2. Push this repository to GitHub.

3. On Render (https://render.com):
   - Create a new **Web Service** and connect your GitHub repo.
   - Render will use `render.yaml` if present; otherwise set:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1`

4. Environment & runtime notes:
   - Render sets `$PORT` automatically. No additional env var is required for basic usage.
   - If you expect many geocoding calls, replace Nominatim with a paid geocoder or add caching to `geocode_address()`.

5. Alternative: use the provided `Dockerfile` to build and push a container instead of using the Python build on Render.

Local smoke-test:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
gunicorn app:app --bind 0.0.0.0:5001 --workers 1
```

That's it — once deployed, POST to `/api/predict` as documented above.

````
# CLAUDE.md - Ypsilanti Rental Price Predictor

## For Every Nick Who Reads This

This is a rental price prediction system for Ypsilanti, Michigan. Think of it as a "Zestimate for renters" - you give it an address and property details, it tells you what the rent should be.

---

## The 30-Second Pitch

**Problem**: You're looking at a rental listing. Is $1,400/month for that 2-bedroom apartment a fair price, or are you getting fleeced?

**Solution**: This project fetches real rental data from the RentCast API, clusters properties into neighborhoods based on location, trains a CatBoost model on the data, and serves predictions via a Flask API.

**One Command to Rule Them All**:
```bash
python3 pipeline.py  # Fetches data, clusters, trains, saves - all in one go
```

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PIPELINE.PY                              │
│  (The Orchestrator - runs every 2-3 months for fresh data)      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌───────────────────┐    ┌─────────────┐  │
│   │  RentCast    │ -> │  K-Means          │ -> │  CatBoost   │  │
│   │  API Fetch   │    │  Clustering       │    │  Training   │  │
│   │  (222 listings)   │  (8 neighborhoods)│    │  (500 iters)│  │
│   └──────────────┘    └───────────────────┘    └─────────────┘  │
│         ↓                      ↓                      ↓         │
│   rentals_rentcast.csv  rentals_with_stats.csv  rental_price_   │
│                                                  model.cbm      │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                          APP.PY                                  │
│  (Flask server - the prediction API)                            │
├─────────────────────────────────────────────────────────────────┤
│   POST /api/predict                                             │
│   {address, beds, baths, sqft, property_type}                   │
│        ↓                                                        │
│   Geocode address → Find nearest cluster → Predict price        │
│        ↓                                                        │
│   {"predicted_price": 1220, "price_range": {low: 978, high: 1462}}│
└─────────────────────────────────────────────────────────────────┘
```

### The "Neighborhood" Trick

Here's the clever bit: location matters for rent, but lat/long coordinates are too granular. A model can't learn that "42.2429, -83.6199" means "downtown Ypsilanti."

**Solution**: K-Means clustering groups properties into 8 neighborhoods based on their coordinates. Each cluster gets a human-readable name via reverse geocoding (Nominatim/OpenStreetMap):

| Cluster ID | Real Name | Count |
|------------|-----------|-------|
| 0 | Midtown | 112 |
| 5 | Miles/Garland | 31 |
| 4 | Ypsilanti Charter Township | 25 |
| 7 | West Willow | 19 |
| 1 | Westlawn | 16 |
| 2 | Gault Village | 10 |
| 3 | Pittsfield Charter Township | 8 |
| 6 | Frain Lake | 1 |

When you predict for a new address, we geocode it, find the nearest cluster centroid, and use that cluster ID as a feature.

---

## Codebase Structure

```
rent/
├── pipeline.py              # The big red button - runs entire workflow
├── app.py                   # Flask API server
├── rental_price_model.py    # Model training, evaluation, prediction utilities
├── rentcast_scraper.py      # Fetches listings from RentCast API
├── cluster_neighborhoods.py # Standalone clustering script (legacy)
├── fetch_neighborhood_stats.py # Walk Score scraper (experimental)
├── zillow_scraper.py        # Zillow scraper (deprecated - they block bots)
│
├── rental_price_model.cbm   # Trained CatBoost model (~577KB)
├── rental_price_model_metadata.json  # Feature names, cluster centroids, CV RMSE
│
├── rentals_rentcast.csv     # Raw listings from last API fetch
├── rentals_with_stats.csv   # Listings with cluster assignments
│
├── backups/                 # Timestamped model backups
│   └── rental_price_model_YYYYMMDD.cbm
├── data/                    # Timestamped raw data
│   └── rentals_raw_YYYYMMDD.csv
│
├── templates/
│   └── index.html           # Web form UI
│
└── catboost_info/           # CatBoost training artifacts (can delete)
```

---

## Technologies Used

| Technology | Why |
|------------|-----|
| **CatBoost** | Handles categorical features natively (property type, cluster ID) without one-hot encoding. Beats XGBoost and Random Forest in our tests. |
| **scikit-learn** | K-Means clustering, cross-validation, train/test split |
| **Flask** | Simple API server. Overkill avoided - no need for Django here. |
| **RentCast API** | Paid API (~$50/month) that provides actual rental listings. Way more reliable than scraping. |
| **Nominatim** | Free reverse geocoding from OpenStreetMap. 1 req/sec rate limit. |
| **pandas** | Data manipulation. The usual suspect. |

---

## Decision Rationale

### Why CatBoost over XGBoost/Random Forest?

We tested all four (see `rental_price_model.py --remove-outliers`):

| Model | CV RMSE |
|-------|---------|
| CatBoost | $239 |
| XGBoost | $248 |
| Random Forest | $261 |
| Linear Regression | $293 |

CatBoost won, plus it handles categorical features without preprocessing gymnastics.

### Why cluster neighborhoods instead of using raw lat/long?

1. **Interpretability**: "Midtown" means something. "42.2429" doesn't.
2. **Generalization**: The model learns "this neighborhood is expensive" rather than overfitting to exact coordinates.
3. **Sparsity**: With only ~220 listings, raw coordinates would be too sparse.

### Why remove outliers?

The IQR method removes ~5% of listings with extreme prices (luxury homes, likely errors). Without this, the model wastes capacity learning outliers instead of the typical rental market.

### Why RentCast instead of scraping Zillow/Craigslist?

Zillow blocks bots aggressively (see `zillow_scraper.py` - it's a graveyard). RentCast costs money but provides clean, structured data with coordinates already included.

---

## Lessons Learned

### Bug #1: numpy.int64 is not JSON serializable

**Symptom**: Pipeline crashes when saving metadata.

**Cause**: `cluster_id = df['Cluster_ID'].unique()[0]` returns `numpy.int64`, not `int`. JSON doesn't know what to do with it.

**Fix**: Always cast: `int(cluster_id)` and `float(cv_rmse)`.

**Lesson**: When saving anything to JSON, wrap numeric values in native Python types.

---

### Bug #2: Geocoding failures cascade

**Symptom**: "Could not geocode address" for valid addresses.

**Cause**: Nominatim is picky. "410 Olive St, Ypsilanti, MI" fails, but "410 Olive Street, Ypsilanti, Michigan" works.

**Fix**: `normalize_address()` tries multiple variations - expanding abbreviations (St → Street), removing unit numbers, expanding state codes.

**Lesson**: When calling external APIs, build retry logic with fallback variations.

---

### Bug #3: Cluster centroids shift between training runs

**Symptom**: Model predicts $800 for an apartment that should be $1,200.

**Cause**: K-Means was initialized with `random_state=None`. Different runs produced different clusters. A new listing would map to cluster 3, but the model was trained when cluster 3 was in a different location.

**Fix**: Always `random_state=42` (or any fixed seed).

**Lesson**: Reproducibility isn't just for papers. It prevents production bugs.

---

### Pitfall: Don't trust "Year Built"

RentCast returns "N/A" for most year_built values. We tried using it as a feature but it hurt performance (the model learned to associate "N/A" with certain prices, which doesn't generalize). We dropped it.

**Lesson**: Missing data isn't just noise - it can be signal of a different distribution. Test both including and excluding sparse features.

---

### Best Practice: Backup before retrain

The pipeline automatically backs up the previous model to `backups/`. This saved us when a bad API response (only 12 listings due to a temporary glitch) produced a garbage model. We rolled back with:

```bash
cp backups/rental_price_model_20260115.cbm rental_price_model.cbm
```

---

### Best Practice: CV RMSE as confidence interval

We expose `cv_rmse` ($239) in the API response as a price range:

```json
{
  "predicted_price": 1220,
  "price_range": {"low": 981, "high": 1459}
}
```

This is more honest than a single number. It says "we're ~68% confident the true price is in this range."

---

## Running the Project

### Full pipeline (every 2-3 months)
```bash
python3 pipeline.py
```

### Start the API server
```bash
python3 app.py
# Server runs at http://localhost:5001
```

### Make a prediction
```bash
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"address": "410 Olive St, Ypsilanti, MI", "beds": 2, "baths": 1, "sqft": 900, "property_type": "Apartment"}'
```

### Retrain model only (using existing data)
```bash
python3 rental_price_model.py --remove-outliers --save
```

---

## API Reference

### POST /api/predict

**Request**:
```json
{
  "address": "410 Olive St, Ypsilanti, MI",
  "beds": 2,
  "baths": 1,
  "sqft": 900,
  "property_type": "Apartment"
}
```

**Response**:
```json
{
  "predicted_price": 1220,
  "price_range": {"low": 981, "high": 1459},
  "neighborhood_cluster": 0,
  "coordinates": {"lat": 42.2476, "lon": -83.618}
}
```

**Property Types**: `Apartment`, `Single Family`, `Condo`, `Townhouse`, `Multi Family`

---

## The Analogy

Think of this system like a real estate agent's intuition, but codified:

1. **Data Collection** (RentCast): The agent drives around, notes prices on signs
2. **Clustering** (K-Means): The agent mentally groups areas - "the college district", "suburban area", "downtown"
3. **Training** (CatBoost): The agent builds intuition - "2-bed apartments downtown go for ~$1,200"
4. **Prediction** (API): A client asks "is $1,400 fair?" The agent says "that's high for the area"

The difference: our agent never forgets, never has a bad day, and processes 220 listings in 30 seconds.

---

## Future Ideas (Not Yet Implemented)

- [ ] Add Walk Score as a feature (see `fetch_neighborhood_stats.py` - works but slow)
- [ ] Historical price trends (requires storing data over months)
- [ ] Compare to asking price and flag overpriced listings
- [ ] Expand beyond Ypsilanti (requires multi-city training)

---

*Last updated: 2026-01-30*
*Model CV RMSE: $239.26*
*Training samples: 211 (after outlier removal)*
