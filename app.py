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
import pandas as pd
from flask import Flask, request, jsonify, render_template
from rental_price_model import load_saved_model, predict_price, geocode_address, find_nearest_cluster
from city_config import CityConfig

app = Flask(__name__)

# Cache for loaded models and data (loaded on first request for each city)
_model_cache = {}
_comps_cache = {}
_metadata_cache = {}


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


# Valid property types (must match training data)
VALID_PROPERTY_TYPES = ['Apartment', 'Single Family', 'Condo', 'Townhouse', 'Multi Family']


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
                                cluster_id: int, comps_df: pd.DataFrame, n_comps: int = 5) -> list:
    """
    Find the N most similar properties from training data.

    Similarity is based on:
    1. Same cluster (neighborhood) - required
    2. Euclidean distance in feature space (beds, baths, sqft normalized)
    """
    if comps_df.empty:
        return []

    df = comps_df.copy()

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
    if property_type not in VALID_PROPERTY_TYPES:
        errors.append(f"Invalid property_type. Must be one of: {VALID_PROPERTY_TYPES}")
    validated['property_type'] = property_type

    if errors:
        return None, "; ".join(errors)

    return validated, None


@app.route('/')
def index():
    """Render the web form UI."""
    return render_template('index.html')


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
        "property_type": "Apartment"
    }

    Response JSON:
    {
        "predicted_price": 1220,
        "price_range": {"low": 978, "high": 1462},
        "city": "ypsilanti",
        "city_name": "Ypsilanti",
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

        # Load model and data for detected city
        model, metadata, comps_df = get_city_model(city_slug)

        # Calculate distance from city center
        distance_from_center = haversine_distance(
            lat, lon,
            city_config.center_lat, city_config.center_lon
        )

        # Find nearest cluster
        centroids = {int(k): v for k, v in metadata['cluster_centroids'].items()}
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

        # Make prediction
        predicted_price = predict_price(
            model=model,
            beds=validated['beds'],
            baths=validated['baths'],
            sqft=validated['sqft'],
            property_type=validated['property_type'],
            cluster_id=cluster_id,
            metadata=metadata
        )

        # Calculate price range using CV RMSE
        cv_rmse = metadata.get('cv_rmse', 242.0)
        price_low = max(0, predicted_price - cv_rmse)
        price_high = predicted_price + cv_rmse

        # Find comparable properties
        comps = find_comparable_properties(
            lat=lat,
            lon=lon,
            beds=validated['beds'],
            baths=validated['baths'],
            sqft=validated['sqft'],
            cluster_id=cluster_id,
            comps_df=comps_df,
            n_comps=5
        )

        return jsonify({
            'predicted_price': round(predicted_price),
            'price_range': {
                'low': round(price_low),
                'high': round(price_high)
            },
            'city': city_slug,
            'city_name': city_config.display_name,
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
