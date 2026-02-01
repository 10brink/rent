"""
app.py - Flask API Server for Rental Price Predictions

The user-facing interface. Loads trained CatBoost models for each city at startup
and serves predictions via REST API. Auto-detects which city a given address is in.

Endpoints:
  GET  /            - Web form UI for manual predictions
  POST /api/predict - JSON API for programmatic access
  GET  /api/cities  - List available cities with trained models

Example:
  curl -X POST http://localhost:5001/api/predict \\
    -H "Content-Type: application/json" \\
    -d '{"address": "410 Olive St, Ypsilanti, MI", "beds": 2, "baths": 1, "sqft": 900, "property_type": "Apartment"}'

Run with: python3 app.py (starts on port 5001)
"""

import os
import math
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from flask import Flask, request, jsonify, render_template
from rental_price_model import load_saved_model, predict_price, geocode_address, find_nearest_cluster
from city_config import CityConfig
from evaluate_comps_knn import compute_comps_metrics

app = Flask(__name__)

# Cache for loaded models and data (loaded on first request for each city)
_model_cache = {}
_comps_cache = {}
_metadata_cache = {}
_comps_metrics_cache = {}
_avg_rents_cache = {}

POOLED_SFH_CITIES = {'east_lansing', 'lansing'}
POOLED_SFH_MODEL_KEY = '__pooled_sfh_lansing_east_lansing'
POOLED_SFH_MODEL_PATH = (
    Path(__file__).parent
    / 'models'
    / 'pools'
    / 'lansing_east_lansing_sfh'
    / 'rental_price_model.cbm'
)


@app.route('/')
def index():
    display_order = ['Ann Arbor', 'Ypsilanti', 'Detroit', 'East Lansing', 'Lansing']
    city_names = []
    for slug in CityConfig.list_trained_cities():
        config = CityConfig(slug)
        city_names.append(config.display_name)
    order_index = {name: idx for idx, name in enumerate(display_order)}
    city_names.sort(key=lambda name: (order_index.get(name, 999), name))
    return render_template('index.html', city_names=city_names)


@app.route('/methodology')
def methodology():
    comps_k = 5
    cities = []
    for slug in CityConfig.list_trained_cities():
        config = CityConfig(slug)
        metadata = get_city_metadata(slug)
        cv_rmse = metadata.get('cv_rmse')
        cv_rmse_display = f"${cv_rmse:,.0f}" if isinstance(cv_rmse, (int, float)) else "n/a"
        comps_metrics = get_comps_metrics(slug, k=comps_k)
        comps_rmse = comps_metrics.get('rmse') if comps_metrics else None
        comps_n = comps_metrics.get('n_predictions') if comps_metrics else None
        comps_rmse_display = f"${comps_rmse:,.0f}" if isinstance(comps_rmse, (int, float)) else "n/a"
        cities.append({
            'slug': slug,
            'name': config.display_name,
            'state': config.state,
            'model_type': metadata.get('model_type', 'CatBoostRegressor'),
            'cv_rmse': cv_rmse,
            'cv_rmse_display': cv_rmse_display,
            'comps_rmse_display': comps_rmse_display,
            'comps_n': comps_n,
            'target_transform': metadata.get('target_transform')
        })
    cities.sort(key=lambda c: c['name'])
    return render_template('methodology.html', cities=cities, comps_k=comps_k)


@app.route('/averages')
def averages():
    city_tables, updated_date = get_avg_rent_tables()
    return render_template('averages.html', city_tables=city_tables, updated_date=updated_date)


def get_city_model(city_slug: str) -> tuple:
    """
    Get model and metadata for a city, loading and caching if needed.

    Returns (model, metadata, comps_df) tuple.
    Raises ValueError if city has no trained model.
    """
    if city_slug in _model_cache:
        return _model_cache[city_slug], _metadata_cache[city_slug], _comps_cache[city_slug]

    config = CityConfig(city_slug)

    if not config.has_trained_model():
        raise ValueError(f"No trained model for {config.display_name}")

    model, metadata = load_saved_model(city=city_slug)
    _model_cache[city_slug] = model
    _metadata_cache[city_slug] = metadata

    # Load training data for comparable properties lookup
    if config.data_path.exists():
        _comps_cache[city_slug] = pd.read_csv(config.data_path)
    else:
        _comps_cache[city_slug] = pd.DataFrame()

    return model, metadata, _comps_cache[city_slug]


def get_city_metadata(city_slug: str) -> dict:
    """Load metadata JSON for a city without loading the full model."""
    if city_slug in _metadata_cache:
        return _metadata_cache[city_slug]

    config = CityConfig(city_slug)
    if not config.metadata_path.exists():
        raise ValueError(f"Missing metadata for {config.display_name}")

    with open(config.metadata_path, 'r') as f:
        metadata = json.load(f)
    _metadata_cache[city_slug] = metadata
    return metadata


def get_comps_metrics(city_slug: str, k: int = 5) -> dict:
    """Compute or retrieve comps-average CV metrics for a city."""
    cache_key = f"{city_slug}:{k}"
    if cache_key in _comps_metrics_cache:
        return _comps_metrics_cache[cache_key]
    try:
        metrics = compute_comps_metrics(city_slug, k=k, property_type_filter=True, min_comps=1)
    except Exception:
        metrics = None
    _comps_metrics_cache[cache_key] = metrics
    return metrics


PROPERTY_TYPE_ORDER = [
    'Apartment',
    'Condo',
    'Townhouse',
    'Single Family',
    'Multi-Family',
]


def normalize_property_type(value: str) -> str:
    """Normalize property type text for consistent grouping."""
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    key = cleaned.lower()
    canonical = {
        'apartment': 'Apartment',
        'condo': 'Condo',
        'single family': 'Single Family',
        'single-family': 'Single Family',
        'multi family': 'Multi-Family',
        'multi-family': 'Multi-Family',
        'multifamily': 'Multi-Family',
        'townhouse': 'Townhouse',
        'townhome': 'Townhouse',
    }
    return canonical.get(key, cleaned)


def format_beds_value(beds: float) -> str:
    """Format beds for display while keeping numeric grouping."""
    if beds is None or pd.isna(beds):
        return None
    if abs(beds - round(beds)) < 1e-6:
        return str(int(round(beds)))
    return f"{beds:g}"


def build_avg_rent_tables(rows: list) -> list:
    """Shape average-rent rows into per-city tables."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row['city'], []).append(row)

    order_index = {name: idx for idx, name in enumerate(PROPERTY_TYPE_ORDER)}
    tables = []

    for city_name, city_rows in grouped.items():
        property_types = sorted(
            {row['property_type'] for row in city_rows},
            key=lambda name: (order_index.get(name, 999), name)
        )
        beds_map = {}
        for row in city_rows:
            label = row['beds']
            value = row.get('_beds_sort')
            if value is None:
                try:
                    value = float(label)
                except (TypeError, ValueError):
                    continue
            beds_map.setdefault(label, value)

        bed_labels = [label for label, _ in sorted(beds_map.items(), key=lambda item: item[1])]

        cells = {}
        for row in city_rows:
            cells.setdefault(row['property_type'], {})[row['beds']] = {
                'avg_price': row['avg_price'],
                'listings': row['listings'],
            }

        tables.append({
            'city': city_name,
            'slug': city_rows[0].get('city_slug'),
            'property_types': property_types,
            'beds': bed_labels,
            'cells': cells,
        })

    tables.sort(key=lambda item: item['city'])
    return tables


def get_avg_rent_rows() -> tuple:
    """Build table rows of average rents by city, property type, and beds."""
    mtimes = {}
    for slug in CityConfig.list_trained_cities():
        config = CityConfig(slug)
        mtimes[slug] = config.data_path.stat().st_mtime if config.data_path.exists() else None

    cache_key = json.dumps(mtimes, sort_keys=True)
    if _avg_rents_cache.get('key') == cache_key:
        return _avg_rents_cache['rows'], _avg_rents_cache.get('updated_date')

    rows = []
    latest_mtime = None

    for slug in CityConfig.list_trained_cities():
        config = CityConfig(slug)
        data_path = config.data_path
        if not data_path.exists():
            continue

        try:
            df = pd.read_csv(data_path, usecols=['Price', 'Beds', 'Property Type'])
        except ValueError:
            df = pd.read_csv(data_path)

        required_cols = {'Price', 'Beds', 'Property Type'}
        if not required_cols.issubset(df.columns):
            continue

        df = df[['Price', 'Beds', 'Property Type']].copy()
        df = df.rename(columns={
            'Price': 'price',
            'Beds': 'beds',
            'Property Type': 'property_type'
        })

        if df['price'].dtype == object:
            df['price'] = df['price'].astype(str).str.replace(r'[$,]', '', regex=True)

        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['beds'] = pd.to_numeric(df['beds'], errors='coerce')
        df['property_type'] = df['property_type'].apply(normalize_property_type)

        df = df.dropna(subset=['price', 'beds', 'property_type'])
        df = df[(df['price'] > 0) & (df['beds'] >= 0)]

        if df.empty:
            continue

        grouped = (
            df.groupby(['property_type', 'beds'])
            .agg(avg_price=('price', 'mean'), listings=('price', 'size'))
            .reset_index()
        )

        for row in grouped.itertuples(index=False):
            beds_value = float(row.beds)
            rows.append({
                'city': f"{config.display_name}, {config.state}",
                'city_slug': slug,
                'property_type': row.property_type,
                'beds': format_beds_value(beds_value),
                'avg_price': round(row.avg_price),
                'listings': int(row.listings),
                '_city_sort': config.display_name.lower(),
                '_beds_sort': beds_value
            })

        file_mtime = data_path.stat().st_mtime
        latest_mtime = file_mtime if latest_mtime is None else max(latest_mtime, file_mtime)

    rows.sort(key=lambda r: (r['_city_sort'], r['property_type'], r['_beds_sort']))
    for row in rows:
        row.pop('_city_sort', None)
        row.pop('_beds_sort', None)

    updated_date = (
        datetime.fromtimestamp(latest_mtime).strftime('%B %d, %Y')
        if latest_mtime else None
    )

    tables = build_avg_rent_tables(rows)

    _avg_rents_cache.update({
        'key': cache_key,
        'rows': rows,
        'tables': tables,
        'updated_date': updated_date
    })
    return rows, updated_date


def get_avg_rent_tables() -> tuple:
    """Return per-city tables and updated date for average rents."""
    rows, updated_date = get_avg_rent_rows()
    tables = _avg_rents_cache.get('tables')
    if tables is None:
        tables = build_avg_rent_tables(rows)
        _avg_rents_cache['tables'] = tables
    return tables, updated_date


def get_pooled_sfh_model() -> tuple:
    """
    Load pooled Single Family model for Lansing + East Lansing.

    Returns (model, metadata) tuple.
    Raises ValueError if the pooled model is missing.
    """
    if POOLED_SFH_MODEL_KEY in _model_cache:
        return _model_cache[POOLED_SFH_MODEL_KEY], _metadata_cache[POOLED_SFH_MODEL_KEY]

    if not POOLED_SFH_MODEL_PATH.exists():
        raise ValueError(
            "Pooled SFH model not found. Train it at "
            f"{POOLED_SFH_MODEL_PATH}"
        )

    model, metadata = load_saved_model(model_path=str(POOLED_SFH_MODEL_PATH))
    _model_cache[POOLED_SFH_MODEL_KEY] = model
    _metadata_cache[POOLED_SFH_MODEL_KEY] = metadata
    return model, metadata


# Valid property types (must match training data)
PROPERTY_TYPE_ALIASES = {
    'Multi Family': 'Multi-Family',
    'Multifamily': 'Multi-Family',
}
VALID_PROPERTY_TYPES = ['Apartment', 'Single Family', 'Condo', 'Townhouse', 'Multi-Family']
VALID_PREDICTION_MODES = ['auto', 'model', 'comps']
DEFAULT_COMPS_K = 5
MIN_COMPS_K = 3
MAX_COMPS_K = 10


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in miles using Haversine formula."""
    R = 3959  # Earth's radius in miles

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def detect_city(lat: float, lon: float) -> tuple:
    """
    Find nearest supported city based on coordinates.

    Returns (city_slug, CityConfig) for the nearest city within range.
    Raises ValueError if address is not within any supported city.
    """
    best_city = None
    best_config = None
    best_distance = float('inf')

    trained_cities = CityConfig.list_trained_cities()

    if not trained_cities:
        raise ValueError("No trained models available. Run pipeline.py to train a model first.")

    for city_slug in trained_cities:
        config = CityConfig(city_slug)
        distance = haversine_distance(lat, lon, config.center_lat, config.center_lon)

        if distance < config.radius_miles and distance < best_distance:
            best_city = city_slug
            best_config = config
            best_distance = distance

    if best_city is None:
        # List available cities in error message
        available = [f"{CityConfig(s).display_name}" for s in trained_cities]
        raise ValueError(
            f"Address is not within any supported city. "
            f"Available cities: {', '.join(available)}"
        )

    return best_city, best_config


def find_comparable_properties(lat: float, lon: float, beds: int, baths: float, sqft: int,
                                cluster_id: int, comps_df: pd.DataFrame, n_comps: int = 5,
                                property_type: str = None) -> list:
    """
    Find the N most similar properties from training data.

    Similarity is based on:
    1. Same cluster (neighborhood) - required
    2. Euclidean distance in feature space (beds, baths, sqft normalized)
    """
    if comps_df.empty:
        return []

    df = comps_df.copy()
    if property_type:
        target_type = str(property_type).strip().lower()
        df = df[df['Property Type'].astype(str).str.lower() == target_type].copy()
        if df.empty:
            return []

    # Filter to same cluster first (neighborhood match is most important for real estate)
    cluster_matches = df[df['Cluster_ID'] == cluster_id]

    # If not enough in same cluster, expand to nearby clusters
    if len(cluster_matches) < n_comps:
        # Use all data but prioritize same cluster
        df['same_cluster'] = (df['Cluster_ID'] == cluster_id).astype(int)
    else:
        df = cluster_matches
        df['same_cluster'] = 1

    # Calculate similarity score (lower is more similar)
    # Normalize features to similar scales
    df['beds_diff'] = abs(df['Beds'] - beds) * 100  # Weight beds heavily
    df['baths_diff'] = abs(df['Baths'] - baths) * 50
    df['sqft_diff'] = abs(df['SqFt'] - sqft) / 10  # Normalize sqft (diff of 100 sqft = 10 points)

    # Geographic distance for tiebreaking
    df['geo_dist'] = df.apply(
        lambda row: haversine_distance(lat, lon, row['Latitude'], row['Longitude']),
        axis=1
    )

    # Combined similarity score (lower = more similar)
    df['similarity'] = df['beds_diff'] + df['baths_diff'] + df['sqft_diff'] + df['geo_dist'] * 10

    # Sort by same_cluster (desc) then similarity (asc)
    df = df.sort_values(['same_cluster', 'similarity'], ascending=[False, True])

    # Remove duplicates by address (keep first/most similar)
    df = df.drop_duplicates(subset=['Address'], keep='first')

    # Get top N comps
    top_comps = df.head(n_comps)

    comps = []
    for _, row in top_comps.iterrows():
        comps.append({
            'address': row['Address'],
            'price': int(row['Price']),
            'beds': int(row['Beds']),
            'baths': float(row['Baths']),
            'sqft': int(row['SqFt']),
            'property_type': row.get('Property Type'),
            'neighborhood': row.get('Neighborhood', f"Cluster {row['Cluster_ID']}"),
            'distance_mi': round(row['geo_dist'], 2)
        })

    return comps


def validate_input(data: dict) -> tuple:
    """
    Validate and sanitize input data.

    Returns (validated_data, error_message).
    If error_message is not None, validation failed.
    """
    errors = []
    validated = {}

    # Required fields
    required_fields = ['address', 'sqft', 'beds', 'baths', 'property_type']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return None, f"Missing required fields: {missing}"

    # Address validation
    address = data.get('address', '').strip()
    if not address:
        errors.append("Address cannot be empty")
    elif len(address) < 5:
        errors.append("Address is too short")
    validated['address'] = address

    # Beds validation (0-10 range)
    try:
        beds = int(data['beds'])
        if beds < 0:
            errors.append("Beds cannot be negative")
        elif beds > 10:
            errors.append("Beds cannot exceed 10")
        validated['beds'] = beds
    except (ValueError, TypeError):
        errors.append("Beds must be a valid integer")

    # Baths validation (0-10 range, allows 0.5 increments)
    try:
        baths = float(data['baths'])
        if baths < 0:
            errors.append("Baths cannot be negative")
        elif baths > 10:
            errors.append("Baths cannot exceed 10")
        validated['baths'] = baths
    except (ValueError, TypeError):
        errors.append("Baths must be a valid number")

    # SqFt validation (100-10000 range)
    try:
        sqft = int(data['sqft'])
        if sqft < 100:
            errors.append("SqFt must be at least 100")
        elif sqft > 10000:
            errors.append("SqFt cannot exceed 10,000")
        validated['sqft'] = sqft
    except (ValueError, TypeError):
        errors.append("SqFt must be a valid integer")

    # Property type validation
    property_type = data.get('property_type', '').strip()
    if property_type in PROPERTY_TYPE_ALIASES:
        property_type = PROPERTY_TYPE_ALIASES[property_type]
    if property_type not in VALID_PROPERTY_TYPES:
        errors.append(f"Invalid property_type. Must be one of: {VALID_PROPERTY_TYPES}")
    validated['property_type'] = property_type

    # Prediction mode validation (optional)
    prediction_mode_raw = data.get('prediction_mode', 'auto')
    prediction_mode = 'auto' if prediction_mode_raw is None else str(prediction_mode_raw).strip().lower()
    if prediction_mode not in VALID_PREDICTION_MODES:
        errors.append(f"Invalid prediction_mode. Must be one of: {VALID_PREDICTION_MODES}")
    validated['prediction_mode'] = prediction_mode

    # Comps override (optional)
    comps_k_raw = data.get('comps_k', DEFAULT_COMPS_K)
    try:
        comps_k = int(comps_k_raw)
        if comps_k < MIN_COMPS_K or comps_k > MAX_COMPS_K:
            errors.append(f"comps_k must be between {MIN_COMPS_K} and {MAX_COMPS_K}")
        validated['comps_k'] = comps_k
    except (ValueError, TypeError):
        errors.append("comps_k must be a valid integer")

    if errors:
        return None, "; ".join(errors)

    return validated, None


@app.route('/api/cities', methods=['GET'])
def api_cities():
    """
    List available cities with trained models.

    Response JSON:
    {
        "cities": [
            {"slug": "ypsilanti", "name": "Ypsilanti", "state": "MI", "trained": true},
            ...
        ]
    }
    """
    cities = []
    for slug in CityConfig.list_cities():
        config = CityConfig(slug)
        cities.append({
            'slug': slug,
            'name': config.display_name,
            'state': config.state,
            'trained': config.has_trained_model()
        })
    return jsonify({'cities': cities})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for rental price predictions.

    Auto-detects city from address coordinates.

    Request JSON:
    {
        "address": "500 E Washington St, Ypsilanti, MI",
        "sqft": 900,
        "beds": 2,
        "baths": 1,
        "property_type": "Apartment",
        "prediction_mode": "auto",  // auto | comps | model
        "comps_k": 5
    }

    Response JSON:
    {
        "predicted_price": 1220,
        "price_range": {"low": 978, "high": 1462},
        "model_prediction": {"price": 1280, "price_range": {"low": 1040, "high": 1520}},
        "comps_prediction": {"price": 1220, "price_range": {"low": 1100, "high": 1350}, "count": 5},
        "prediction_order": ["model", "comps"],
        "city": "ypsilanti",
        "city_name": "Ypsilanti",
        "prediction_mode_used": "model",
        "neighborhood": "Midtown",
        "coordinates": {"lat": 42.2803, "lon": -83.7431},
        "comparable_properties": [
            {"address": "...", "price": 1150, "beds": 2, "baths": 1, "sqft": 875, ...},
            ...
        ]
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    # Validate input
    validated, error = validate_input(data)
    if error:
        return jsonify({'error': error}), 400

    try:
        # Geocode the address
        lat, lon = geocode_address(validated['address'])

        # Auto-detect city from coordinates
        city_slug, city_config = detect_city(lat, lon)

        # Load model and data for detected city (used for centroids/comps)
        city_model, city_metadata, comps_df = get_city_model(city_slug)

        prediction_mode = validated.get('prediction_mode', 'auto')
        comps_k = validated.get('comps_k', DEFAULT_COMPS_K)

        # KNN comps for East Lansing single-family homes (auto mode)
        use_comps_knn = (
            prediction_mode == 'comps'
            or (
                prediction_mode == 'auto'
                and validated['property_type'] == 'Single Family'
                and city_slug == 'east_lansing'
            )
        )

        # Optionally use pooled SFH model for Lansing + East Lansing
        use_pooled_sfh = (
            validated['property_type'] == 'Single Family'
            and city_slug in POOLED_SFH_CITIES
            and not use_comps_knn
        )
        model = city_model
        model_metadata = city_metadata
        extra_features = None
        if use_pooled_sfh:
            try:
                model, model_metadata = get_pooled_sfh_model()
                extra_features = {'City_Slug': city_slug}
            except ValueError:
                # Fall back to city-specific model if pooled model missing
                model = city_model
                model_metadata = city_metadata
                extra_features = None

        # Calculate distance from city center
        distance_from_center = haversine_distance(
            lat, lon,
            city_config.center_lat, city_config.center_lon
        )

        # Find nearest cluster
        centroids = {int(k): v for k, v in city_metadata['cluster_centroids'].items()}
        cluster_id = find_nearest_cluster(lat, lon, centroids)

        # Get neighborhood name (from metadata, or fall back to training data)
        neighborhood_name = centroids[cluster_id].get('name')
        if not neighborhood_name:
            # Look up from training data
            if not comps_df.empty:
                cluster_rows = comps_df[comps_df['Cluster_ID'] == cluster_id]
                if len(cluster_rows) > 0 and 'Neighborhood' in cluster_rows.columns:
                    neighborhood_name = cluster_rows['Neighborhood'].iloc[0]
                else:
                    neighborhood_name = f'Cluster {cluster_id}'
            else:
                neighborhood_name = f'Cluster {cluster_id}'

        # Find comparable properties
        comps = find_comparable_properties(
            lat=lat,
            lon=lon,
            beds=validated['beds'],
            baths=validated['baths'],
            sqft=validated['sqft'],
            cluster_id=cluster_id,
            comps_df=comps_df,
            n_comps=comps_k,
            property_type=validated['property_type'] if use_comps_knn else None
        )

        # Build comps prediction (if available)
        comp_prices = [c['price'] for c in comps if isinstance(c.get('price'), (int, float))]
        comps_avg = None
        comps_low = None
        comps_high = None
        comps_count = 0
        if comp_prices:
            comps_count = len(comp_prices)
            comps_avg = sum(comp_prices) / comps_count
            comps_low = min(comp_prices)
            comps_high = max(comp_prices)
        elif prediction_mode == 'comps':
            return jsonify({'error': 'No comparable properties found for comps-average mode. Try model mode.'}), 400

        # Always compute model prediction for display
        model_prediction = predict_price(
            model=model,
            beds=validated['beds'],
            baths=validated['baths'],
            sqft=validated['sqft'],
            property_type=validated['property_type'],
            cluster_id=cluster_id,
            metadata=model_metadata,
            extra_features=extra_features
        )

        # Calculate model price range using CV RMSE
        cv_rmse = model_metadata.get('cv_rmse', 242.0)
        model_low = max(0, model_prediction - cv_rmse)
        model_high = model_prediction + cv_rmse

        comps_only_rule_applied = (
            prediction_mode == 'auto'
            and validated['property_type'] == 'Single Family'
            and city_slug == 'east_lansing'
        )

        used_comps_average = use_comps_knn and comps_avg is not None
        primary_mode = 'comps' if used_comps_average else 'model'

        if primary_mode == 'comps':
            predicted_price = comps_avg
            price_low = comps_low
            price_high = comps_high
        else:
            predicted_price = model_prediction
            price_low = model_low
            price_high = model_high

        prediction_order = ['model', 'comps']
        if comps_only_rule_applied and comps_avg is not None:
            prediction_order = ['comps', 'model']
        if comps_avg is None:
            prediction_order = [mode for mode in prediction_order if mode != 'comps']

        return jsonify({
            'predicted_price': round(predicted_price),
            'price_range': {
                'low': round(price_low),
                'high': round(price_high)
            },
            'model_prediction': {
                'price': round(model_prediction),
                'price_range': {
                    'low': round(model_low),
                    'high': round(model_high)
                }
            },
            'comps_prediction': None if comps_avg is None else {
                'price': round(comps_avg),
                'price_range': {
                    'low': round(comps_low),
                    'high': round(comps_high)
                },
                'count': comps_count
            },
            'prediction_order': prediction_order,
            'city': city_slug,
            'city_name': city_config.display_name,
            'state': city_config.state,
            'comps_k': comps_k,
            'prediction_mode_requested': prediction_mode,
            'prediction_mode_used': primary_mode,
            'neighborhood': neighborhood_name,
            'coordinates': {
                'lat': round(lat, 4),
                'lon': round(lon, 4)
            },
            'distance_from_center_mi': round(distance_from_center, 2),
            'comparable_properties': comps
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
