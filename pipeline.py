"""
pipeline.py - The Big Red Button

Automates the complete data collection and model training workflow.
Run this every 2-3 months to refresh the model with current rental data.

Steps:
  1. Backup existing model to models/{city}/backups/
  2. Fetch fresh listings from RentCast API (with pagination)
  3. Cluster properties into neighborhoods via K-Means
  4. Reverse geocode clusters to get real neighborhood names
  5. Train CatBoost model with outlier removal
  6. Save model and metadata

Usage:
  python3 pipeline.py --city ypsilanti      # Full pipeline for Ypsilanti
  python3 pipeline.py --city ann_arbor      # Full pipeline for Ann Arbor
  python3 pipeline.py --city detroit        # Full pipeline for Detroit
  python3 pipeline.py --dry-run --city ann_arbor  # Fetch data only, don't train
  python3 pipeline.py --skip-backup --city ypsilanti  # Don't backup previous model
  python3 pipeline.py --append --city ann_arbor  # Add new listings to existing data
"""

import argparse
import json
import os
import random
from dotenv import load_dotenv

load_dotenv('.env.local')
import shutil
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from city_config import CityConfig
from rental_price_model import clean_outliers, impute_missing_values, compute_city_medians


def get_timestamp() -> str:
    """Return current timestamp in YYYYMMDD format."""
    return datetime.now().strftime("%Y%m%d")


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """
    Reverse geocode coordinates to get neighborhood/suburb name.

    Uses Nominatim (OpenStreetMap) API.
    Returns the most specific locality name found.
    """
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&zoom=14"

    req = urllib.request.Request(url, headers={'User-Agent': 'RentalPricePredictor/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode(errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(
            f"Nominatim HTTP {exc.code} for lat={lat}, lon={lon}. {body}".strip()
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Nominatim request failed for lat={lat}, lon={lon}: {exc.reason}"
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"Failed to decode Nominatim response for lat={lat}, lon={lon}"
        ) from exc

    address = data.get('address', {})

    # Try to get the most descriptive local name, in order of preference
    for key in ['neighbourhood', 'suburb', 'hamlet', 'village', 'town', 'city_district']:
        if key in address:
            return address[key]

    # Fallback to city or county
    return address.get('city', address.get('county', 'Unknown'))


class KMeansClusterer(BaseEstimator, TransformerMixin):
    """Fit KMeans on lat/lon in the training fold and assign Cluster_ID."""

    def __init__(self, n_clusters: int, random_state: int = 42, n_init: int = 10) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.kmeans_ = None

    def fit(self, X: pd.DataFrame, y=None):
        coords = X[['Latitude', 'Longitude']].copy()
        coords['Latitude'] = pd.to_numeric(coords['Latitude'], errors='coerce')
        coords['Longitude'] = pd.to_numeric(coords['Longitude'], errors='coerce')
        n_samples = len(coords)
        if n_samples == 0:
            self.kmeans_ = None
            self.effective_n_clusters_ = 0
            return self
        self.effective_n_clusters_ = min(self.n_clusters, n_samples)
        self.kmeans_ = KMeans(
            n_clusters=self.effective_n_clusters_,
            random_state=self.random_state,
            n_init=self.n_init
        )
        self.kmeans_.fit(coords.values)
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if self.kmeans_ is None:
            X['Cluster_ID'] = 0
            return X
        coords = X[['Latitude', 'Longitude']].copy()
        coords['Latitude'] = pd.to_numeric(coords['Latitude'], errors='coerce')
        coords['Longitude'] = pd.to_numeric(coords['Longitude'], errors='coerce')
        X['Cluster_ID'] = self.kmeans_.predict(coords.values)
        return X


class ImputerByCity(BaseEstimator, TransformerMixin):
    """Impute Beds/Baths/SqFt using medians learned from the training fold."""

    def __init__(self) -> None:
        self.city_medians_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X_num = X.copy()
        for col in ['Beds', 'Baths', 'SqFt']:
            if col in X_num.columns:
                X_num[col] = pd.to_numeric(X_num[col], errors='coerce')
        self.city_medians_ = compute_city_medians(X_num)
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in ['Beds', 'Baths', 'SqFt']:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        return impute_missing_values(X, city_medians=self.city_medians_)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select model feature columns and coerce Cluster_ID to string."""

    def __init__(self, feature_cols: List[str]) -> None:
        self.feature_cols = feature_cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if 'Cluster_ID' in X.columns:
            X['Cluster_ID'] = X['Cluster_ID'].astype(str)
        existing = [c for c in self.feature_cols if c in X.columns]
        return X[existing]


def get_neighborhood_names(kmeans, df: pd.DataFrame, city_config: CityConfig) -> dict:
    """
    Get descriptive names for each cluster by reverse geocoding centroids.

    Returns dict mapping cluster_id to neighborhood name.
    """
    centers = kmeans.cluster_centers_
    cluster_names = {}
    used_names = set()

    print("Reverse geocoding cluster centroids...")

    for cluster_id in range(len(centers)):
        lat, lon = centers[cluster_id]
        name = reverse_geocode(lat, lon)

        if name and name not in used_names:
            cluster_names[cluster_id] = name
            used_names.add(name)
        else:
            # Generate a directional name based on position relative to city center
            city_center_lat = city_config.center_lat
            city_center_lon = city_config.center_lon

            lat_diff = lat - city_center_lat
            lon_diff = lon - city_center_lon

            if abs(lat_diff) > abs(lon_diff):
                direction = "North" if lat_diff > 0 else "South"
            else:
                direction = "East" if lon_diff > 0 else "West"

            # Add distance qualifier
            dist = ((lat_diff ** 2) + (lon_diff ** 2)) ** 0.5
            if dist > 0.05:
                direction = f"Far {direction}"

            fallback_name = f"{direction} {city_config.display_name}"

            # Ensure uniqueness
            counter = 1
            unique_name = fallback_name
            while unique_name in used_names:
                counter += 1
                unique_name = f"{fallback_name} {counter}"

            cluster_names[cluster_id] = unique_name
            used_names.add(unique_name)

        # Rate limit for Nominatim API (1 request per second)
        time.sleep(1)

    return cluster_names


def backup_model(city_config: CityConfig, skip_backup: bool = False) -> dict:
    """
    Backup existing model and metadata files.

    Returns dict with backup info or empty dict if no backup made.
    """
    if skip_backup:
        print("Skipping model backup (--skip-backup)")
        return {}

    model_path = city_config.model_path
    metadata_path = city_config.metadata_path

    if not model_path.exists():
        print("No existing model found, skipping backup")
        return {}

    timestamp = get_timestamp()
    backup_dir = city_config.backup_dir
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_model_path = backup_dir / f"rental_price_model_{timestamp}.cbm"
    backup_metadata_path = backup_dir / f"rental_price_model_metadata_{timestamp}.json"

    # Check if backup already exists for today
    if backup_model_path.exists():
        print(f"Backup already exists for today: {backup_model_path}")
        return {"model": str(backup_model_path), "metadata": str(backup_metadata_path)}

    print(f"Backing up model to {backup_model_path}")
    shutil.copy2(model_path, backup_model_path)

    if metadata_path.exists():
        shutil.copy2(metadata_path, backup_metadata_path)
        print(f"Backed up metadata to {backup_metadata_path}")

    # Load old CV RMSE for comparison
    old_cv_rmse = None
    if metadata_path.exists():
        with open(metadata_path) as f:
            old_meta = json.load(f)
            old_cv_rmse = old_meta.get("cv_rmse")

    return {
        "model": str(backup_model_path),
        "metadata": str(backup_metadata_path),
        "old_cv_rmse": old_cv_rmse
    }


def fetch_rentcast_data(city_config: CityConfig, max_pages: int = 10, append: bool = False) -> tuple:
    """
    Fetch rental listings from RentCast API with pagination.

    Args:
        city_config: CityConfig instance for the city to fetch
        max_pages: Maximum number of API pages to fetch (500 listings each)
        append: If True, merge with existing data instead of replacing

    Saves timestamped copy to data/{city}/raw/ and returns the DataFrame.
    Returns tuple of (DataFrame, metadata dict).
    """
    api_key = os.getenv('RENTCAST_API_KEY')
    if not api_key:
        raise ValueError("RENTCAST_API_KEY environment variable not set")

    url = "https://api.rentcast.io/v1/listings/rental/long-term"

    headers = {
        "accept": "application/json",
        "X-Api-Key": api_key
    }

    city_config.ensure_directories()

    request_timeout = (5, 30)  # (connect, read) seconds
    max_retries = 5
    backoff_base = 1.0
    backoff_max = 30.0
    retryable_statuses = {429, 500, 502, 503, 504}

    print(f"\nFetching listings from RentCast API for {city_config.display_name}, {city_config.state}...")
    print(f"  Max pages: {max_pages} (up to {max_pages * 500} listings)")
    print(
        f"  Request timeout: {request_timeout[0]}s connect / {request_timeout[1]}s read, "
        f"max retries: {max_retries}"
    )

    all_listings = []
    pages_fetched = 0

    for page in range(1, max_pages + 1):
        params = {
            "city": city_config.display_name,
            "state": city_config.state,
            "status": "Active",
            "limit": 500,
            "offset": (page - 1) * 500
        }

        attempt = 0
        while True:
            try:
                response = requests.get(url, headers=headers, params=params, timeout=request_timeout)
                if response.status_code in retryable_statuses:
                    raise requests.exceptions.HTTPError(
                        f"Retryable HTTP {response.status_code}",
                        response=response
                    )
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else None
                if status_code not in retryable_statuses:
                    raise
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(
                        f"RentCast request failed for page {page} after {max_retries} retries "
                        f"(last status: {status_code})"
                    ) from e
                retry_after = None
                if status_code == 429 and e.response is not None:
                    header_value = e.response.headers.get("Retry-After")
                    if header_value:
                        try:
                            retry_after = int(header_value)
                        except ValueError:
                            retry_after = None
                sleep_seconds = retry_after if retry_after is not None else min(
                    backoff_base * (2 ** (attempt - 1)), backoff_max
                )
                sleep_seconds = sleep_seconds + random.random()
                print(
                    f"  Page {page}: HTTP {status_code}, retrying in {sleep_seconds:.1f}s "
                    f"(attempt {attempt}/{max_retries})"
                )
                time.sleep(sleep_seconds)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(
                        f"RentCast request failed for page {page} after {max_retries} retries "
                        f"({e.__class__.__name__})"
                    ) from e
                sleep_seconds = min(backoff_base * (2 ** (attempt - 1)), backoff_max)
                sleep_seconds = sleep_seconds + random.random()
                print(
                    f"  Page {page}: {e.__class__.__name__}, retrying in {sleep_seconds:.1f}s "
                    f"(attempt {attempt}/{max_retries})"
                )
                time.sleep(sleep_seconds)

        data = response.json()
        listings = data if isinstance(data, list) else data.get("listings", [])

        if not listings:
            print(f"  Page {page}: No more listings")
            break

        all_listings.extend(listings)
        pages_fetched = page
        print(f"  Page {page}: {len(listings)} listings (total: {len(all_listings)})")

        # If we got fewer than 500, we've reached the end
        if len(listings) < 500:
            break

    if not all_listings:
        raise ValueError(f"No listings found from RentCast API for {city_config.display_name}")

    print(f"Found {len(all_listings)} total listings across {pages_fetched} pages")

    def _norm_part(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip().lower()

    def _dedupe_key_from_row(row: pd.Series) -> str:
        listing_id = row.get("Listing ID")
        if listing_id is not None and str(listing_id).strip() != "":
            return f"id:{str(listing_id).strip().lower()}"

        parts = [
            _norm_part(row.get("Address")),
            _norm_part(row.get("Address 2")),
            _norm_part(row.get("City")),
            _norm_part(row.get("State")),
            _norm_part(row.get("Zip"))
        ]
        if not any(parts):
            parts = [_norm_part(row.get("Formatted Address"))]
        return "addr:" + "|".join(parts)

    processed_data = []
    for item in all_listings:
        listing_id = item.get("id") or item.get("listingId") or item.get("listing_id")
        address_line2 = item.get("addressLine2") or item.get("unit") or item.get("unitNumber") or item.get("apartment")
        city = item.get("city") or item.get("addressCity") or city_config.display_name
        state = item.get("state") or item.get("addressState") or city_config.state
        zip_code = item.get("zipCode") or item.get("postalCode") or item.get("zip")
        formatted_address = item.get("formattedAddress")
        listed_date = item.get("listedDate")
        removed_date = item.get("removedDate")
        created_date = item.get("createdDate")
        last_seen_date = item.get("lastSeenDate")
        days_on_market = item.get("daysOnMarket")
        days_old = item.get("daysOld")
        processed_data.append({
            "Listing ID": listing_id,
            "Address": item.get("addressLine1", "N/A"),
            "Address 2": address_line2,
            "City": city,
            "State": state,
            "Zip": zip_code,
            "Formatted Address": formatted_address,
            "Listed Date": listed_date,
            "Removed Date": removed_date,
            "Created Date": created_date,
            "Last Seen Date": last_seen_date,
            "Days On Market": days_on_market,
            "Days Old": days_old,
            "Price": item.get("price", "N/A"),
            "Beds": item.get("bedrooms", "N/A"),
            "Baths": item.get("bathrooms", "N/A"),
            "SqFt": item.get("squareFootage", "N/A"),
            "Latitude": item.get("latitude", "N/A"),
            "Longitude": item.get("longitude", "N/A"),
            "Property Type": item.get("propertyType", "N/A"),
            "Year Built": item.get("yearBuilt", "N/A"),
            "Amenities": ", ".join(item.get("features", [])) or "N/A"
        })

    df = pd.DataFrame(processed_data)
    # Normalize missing values and ensure numeric types for model features
    df = df.replace({"N/A": np.nan, "": np.nan})
    for col in ["Price", "Beds", "Baths", "SqFt", "Latitude", "Longitude", "Year Built"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save timestamped version
    timestamp = get_timestamp()
    timestamped_path = city_config.raw_data_dir / f"rentals_raw_{timestamp}.csv"
    df.to_csv(timestamped_path, index=False)
    print(f"Saved timestamped data to {timestamped_path}")

    # Handle append mode
    if append and city_config.data_path.exists():
        existing_df = pd.read_csv(city_config.data_path)
        # Merge on Listing ID if present; otherwise use full address parts
        existing_keys = set(existing_df.apply(_dedupe_key_from_row, axis=1))
        incoming_keys = df.apply(_dedupe_key_from_row, axis=1)
        new_rows = df[~incoming_keys.isin(existing_keys)]
        print(f"Append mode: {len(new_rows)} new listings (skipping {len(df) - len(new_rows)} duplicates)")
        df = pd.concat([existing_df, new_rows], ignore_index=True)
        # Re-apply numeric coercion after concat (pandas may upcast to object)
        for col in ["Price", "Beds", "Baths", "SqFt", "Latitude", "Longitude", "Year Built"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build metadata about this fetch
    fetch_metadata = {
        "fetch_date": datetime.now().isoformat(),
        "city": city_config.display_name,
        "state": city_config.state,
        "total_listings": len(df),
        "pages_fetched": pages_fetched,
        "append_mode": append,
        "request_timeout_seconds": {
            "connect": request_timeout[0],
            "read": request_timeout[1]
        },
        "max_retries": max_retries,
        "backoff_base_seconds": backoff_base,
        "backoff_max_seconds": backoff_max,
        "retryable_statuses": sorted(retryable_statuses)
    }

    return df, fetch_metadata


def add_clusters(df: pd.DataFrame, city_config: CityConfig, verbose: bool = False) -> tuple:
    """
    Add neighborhood clusters based on lat/long coordinates.

    Saves output as data/{city}/rentals_with_stats.csv.
    Returns tuple of (clustered DataFrame, neighborhood_names dict).
    """
    n_clusters = city_config.n_clusters
    print(f"\nAdding neighborhood clusters (n={n_clusters}) for {city_config.display_name}...")

    # Convert coordinates to numeric
    df = df.copy()
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    initial_count = len(df)
    df_clean = df.dropna(subset=['Latitude', 'Longitude']).copy()
    dropped = initial_count - len(df_clean)

    if len(df_clean) < n_clusters:
        raise ValueError(f"Not enough data points ({len(df_clean)}) for {n_clusters} clusters")

    print(f"Clustering {len(df_clean)} records (dropped {dropped} with missing coordinates)")

    # Perform K-Means clustering
    coords = df_clean[['Latitude', 'Longitude']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clean['Cluster_ID'] = kmeans.fit_predict(coords)

    # Get descriptive neighborhood names via reverse geocoding
    neighborhood_names = get_neighborhood_names(kmeans, df_clean, city_config)
    df_clean['Neighborhood'] = df_clean['Cluster_ID'].apply(
        lambda x: neighborhood_names.get(x, f"Neighborhood {x + 1}")
    )

    # Clean impossible values and impute missing data (single point of transformation)
    df_clean = clean_outliers(df_clean, verbose=verbose)
    # Compute city-specific medians for imputation (after outlier cleanup)
    city_medians = compute_city_medians(df_clean)
    if verbose:
        medians_display = ", ".join(
            f"{beds}:{sqft}" for beds, sqft in sorted(city_medians.items())
        )
        print(f"City medians (Beds->SqFt): {medians_display}")
    df_clean = impute_missing_values(df_clean, city_medians=city_medians, verbose=verbose)

    # Save output (now clean and ready for training)
    city_config.ensure_directories()
    df_clean.to_csv(city_config.data_path, index=False)
    print(f"Saved clustered data to {city_config.data_path}")

    # Print cluster summary
    print(f"\nCluster distribution:")
    for cluster_id, name in sorted(neighborhood_names.items()):
        count = (df_clean['Cluster_ID'] == cluster_id).sum()
        print(f"  {name}: {count} listings")

    return df_clean, neighborhood_names


def remove_outliers_iqr(X: pd.DataFrame, y: pd.Series, multiplier: float = 1.5) -> tuple:
    """Remove price outliers using IQR method."""
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    mask = (y >= lower_bound) & (y <= upper_bound)
    n_removed = (~mask).sum()

    stats = {
        'n_original': len(y),
        'n_removed': n_removed,
        'n_remaining': mask.sum(),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), stats


def compute_cluster_centroids(df: pd.DataFrame, neighborhood_names: dict = None) -> dict:
    """Compute cluster centroids from the data, including neighborhood names."""
    centroids = {}
    for cid in df['Cluster_ID'].unique():
        subset = df[df['Cluster_ID'] == cid]
        # Use int() to convert numpy.int64 to native Python int
        cluster_id = int(cid)
        centroids[cluster_id] = {
            'latitude': float(subset['Latitude'].mean()),
            'longitude': float(subset['Longitude'].mean()),
            'count': int(len(subset)),
            'name': neighborhood_names.get(cluster_id, f"Neighborhood {cluster_id + 1}") if neighborhood_names else f"Neighborhood {cluster_id + 1}"
        }
    return centroids


def train_and_save_model(df: pd.DataFrame, city_config: CityConfig,
                         neighborhood_names: dict = None, dry_run: bool = False,
                         df_for_cv: Optional[pd.DataFrame] = None,
                         verbose: bool = False) -> dict:
    """
    Train CatBoost model on the data and save it.

    Returns dict with training statistics.
    """
    print(f"\nPreparing training data for {city_config.display_name}...")

    def _fmt_price(value: float) -> str:
        return "N/A" if pd.isna(value) else f"${value:,.0f}"

    def _skip_training(reason: str, outlier_stats: dict, sample_count: int) -> dict:
        print(f"[SKIP] {reason}")
        return {
            'n_samples': sample_count,
            'n_outliers_removed': outlier_stats['n_removed'],
            'cv_rmse': None,
            'dry_run': dry_run,
            'skipped_training': True,
            'skip_reason': reason
        }

    # Prepare features
    feature_cols = ['Beds', 'Baths', 'SqFt', 'Property Type', 'Cluster_ID']
    available_cols = [col for col in feature_cols if col in df.columns]

    X = df[available_cols].copy()
    y_raw = df['Price'].copy()

    # Convert Cluster_ID to string for categorical handling
    if 'Cluster_ID' in X.columns:
        X['Cluster_ID'] = X['Cluster_ID'].astype(str)

    # Remove outliers
    X, y_raw, outlier_stats = remove_outliers_iqr(X, y_raw)
    print(f"Removed {outlier_stats['n_removed']} outliers (kept {outlier_stats['n_remaining']})")
    print(f"Price bounds: {_fmt_price(outlier_stats['lower_bound'])} - {_fmt_price(outlier_stats['upper_bound'])}")
    if verbose:
        n_original = int(outlier_stats['n_original'])
        n_removed = int(outlier_stats['n_removed'])
        n_remaining = int(outlier_stats['n_remaining'])
        pct_removed = (n_removed / n_original * 100) if n_original else 0.0
        print(
            f"Outlier removal: removed {n_removed}/{n_original} "
            f"({pct_removed:.1f}%), remaining {n_remaining}"
        )

    n_remaining = int(outlier_stats['n_remaining'])
    min_training_samples = 20
    if n_remaining == 0 or y_raw.isna().all():
        reason = "no valid prices remain after outlier removal"
        return _skip_training(reason, outlier_stats, n_remaining)
    if n_remaining < min_training_samples:
        reason = f"only {n_remaining} samples after outlier removal (min {min_training_samples})"
        return _skip_training(reason, outlier_stats, n_remaining)

    cat_features = ['Property Type', 'Cluster_ID']
    cat_features = [f for f in cat_features if f in X.columns]

    print(f"\nTraining CatBoost model...")
    print(f"  Features: {list(X.columns)}")
    print(f"  Samples: {len(X)}")
    print(f"  Price range: {_fmt_price(y_raw.min())} - {_fmt_price(y_raw.max())}")
    print("  Target transform: log1p(price)")

    if dry_run:
        print("\n[DRY RUN] Skipping model training and saving")
        return {
            'n_samples': len(X),
            'n_outliers_removed': outlier_stats['n_removed'],
            'cv_rmse': None,
            'dry_run': True
        }

    y_log = np.log1p(y_raw)

    # Train model
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,
        random_state=42,
        verbose=100
    )
    model.fit(X, y_log)

    # Cross-validation with leakage-safe pipeline (skip outlier removal during CV)
    print("\nRunning 5-fold cross-validation (leakage-safe pipeline)...")
    df_cv = df_for_cv.copy() if df_for_cv is not None else df.copy()
    df_cv = df_cv.replace({"N/A": np.nan, "": np.nan})
    for col in ["Price", "Beds", "Baths", "SqFt", "Latitude", "Longitude"]:
        if col in df_cv.columns:
            df_cv[col] = pd.to_numeric(df_cv[col], errors="coerce")

    required_cols = ["Beds", "Baths", "SqFt", "Property Type", "Latitude", "Longitude", "Price"]
    cv_before = len(df_cv)
    df_cv = df_cv.dropna(subset=["Latitude", "Longitude", "Price"])
    cv_after = len(df_cv)
    if verbose:
        dropped = cv_before - cv_after
        pct_dropped = (dropped / cv_before * 100) if cv_before else 0.0
        print(
            f"CV data: {cv_after}/{cv_before} rows after dropping missing "
            f"lat/lon/price ({pct_dropped:.1f}% removed)"
        )
    X_cv = df_cv[[c for c in required_cols if c != "Price"]].copy()
    y_cv = df_cv["Price"].copy()

    feature_cols = ['Beds', 'Baths', 'SqFt', 'Property Type', 'Cluster_ID']
    cv_cat_features = [f for f in ['Property Type', 'Cluster_ID'] if f in feature_cols]

    base_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        cat_features=cv_cat_features,
        random_state=42,
        verbose=0
    )
    reg = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,
        inverse_func=np.expm1
    )

    cv_pipeline = Pipeline(steps=[
        ("cluster", KMeansClusterer(n_clusters=city_config.n_clusters, random_state=42, n_init=10)),
        ("impute", ImputerByCity()),
        ("select", FeatureSelector(feature_cols)),
        ("model", reg),
    ])

    def _log_rmse(y_true, y_pred) -> float:
        y_pred = np.maximum(y_pred, 0)
        return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))

    rmse_scores = cross_val_score(
        cv_pipeline, X_cv, y_cv,
        cv=5, scoring='neg_mean_squared_error'
    )
    cv_rmse = float(np.sqrt(-rmse_scores).mean())

    log_rmse_scores = cross_val_score(
        cv_pipeline, X_cv, y_cv,
        cv=5, scoring=make_scorer(_log_rmse, greater_is_better=False)
    )
    cv_rmse_log = float(-log_rmse_scores.mean())
    print(f"CV RMSE: ${cv_rmse:,.2f}")

    # Time-based holdout split using Listed Date (train on older, validate on newer)
    time_rmse = None
    time_rmse_log = None
    time_split_date = None
    if "Listed Date" in df_cv.columns:
        df_time = df_cv.copy()
        df_time["Listed Date"] = pd.to_datetime(df_time["Listed Date"], errors="coerce", utc=True)
        df_time = df_time.dropna(subset=["Listed Date"])
        df_time = df_time.sort_values("Listed Date")
        if len(df_time) >= 20:
            split_idx = int(len(df_time) * 0.8)
            if 0 < split_idx < len(df_time):
                time_split_date = df_time["Listed Date"].iloc[split_idx]
                X_time = df_time[[c for c in required_cols if c != "Price"]].copy()
                y_time = df_time["Price"].copy()

                X_train_time = X_time.iloc[:split_idx]
                y_train_time = y_time.iloc[:split_idx]
                X_val_time = X_time.iloc[split_idx:]
                y_val_time = y_time.iloc[split_idx:]

                cv_pipeline.fit(X_train_time, y_train_time)
                preds_time = cv_pipeline.predict(X_val_time)
                preds_time = np.maximum(preds_time, 0)

                time_rmse = float(np.sqrt(mean_squared_error(y_val_time, preds_time)))
                time_rmse_log = float(_log_rmse(y_val_time, preds_time))
                print(f"Time-split RMSE: ${time_rmse:,.2f} (cutoff: {time_split_date.date()})")

    # Compute cluster centroids with neighborhood names
    centroids = compute_cluster_centroids(df, neighborhood_names)

    # Save model
    city_config.ensure_directories()
    model.save_model(str(city_config.model_path))
    print(f"\nModel saved to: {city_config.model_path}")

    # Save metadata (ensure all values are JSON-serializable)
    metadata = {
        'feature_names': list(X.columns),
        'categorical_features': cat_features,
        'model_type': 'CatBoostRegressor',
        'model_file': str(city_config.model_path),
        'cluster_centroids': centroids,
        'cv_rmse': float(cv_rmse),
        'cv_rmse_log': float(cv_rmse_log),
        'target_transform': 'log1p',
        'time_rmse': float(time_rmse) if time_rmse is not None else None,
        'time_rmse_log': float(time_rmse_log) if time_rmse_log is not None else None,
        'time_split_date': time_split_date.isoformat() if time_split_date is not None else None,
        'training_date': get_timestamp(),
        'n_samples': int(len(X)),
        'n_outliers_removed': int(outlier_stats['n_removed']),
        'city': city_config.slug,
        'city_display_name': city_config.display_name,
        'city_state': city_config.state,
        'city_center': {
            'lat': city_config.center_lat,
            'lon': city_config.center_lon
        },
        'city_radius_miles': city_config.radius_miles
    }

    with open(city_config.metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {city_config.metadata_path}")

    return {
        'n_samples': len(X),
        'n_outliers_removed': outlier_stats['n_removed'],
        'cv_rmse': cv_rmse,
        'dry_run': False
    }


def print_summary(city_config: CityConfig, backup_info: dict, fetch_count: int,
                  cluster_count: int, training_stats: dict, fetch_metadata: dict = None) -> None:
    """Print pipeline execution summary."""
    print("\n" + "=" * 60)
    print(f"PIPELINE SUMMARY - {city_config.display_name}, {city_config.state}")
    print("=" * 60)

    print(f"\nData Collection:")
    print(f"  Records fetched: {fetch_count}")
    print(f"  Records clustered: {cluster_count}")
    if fetch_metadata:
        print(f"  Pages fetched: {fetch_metadata.get('pages_fetched', 'N/A')}")
        print(f"  Fetch date: {fetch_metadata.get('fetch_date', 'N/A')}")

    print(f"\nModel Training:")
    print(f"  Training samples: {training_stats['n_samples']}")
    print(f"  Outliers removed: {training_stats['n_outliers_removed']}")

    if training_stats.get('skipped_training'):
        reason = training_stats.get('skip_reason', 'insufficient data')
        print(f"  [SKIPPED - {reason}]")
    elif training_stats.get('dry_run'):
        print("  [DRY RUN - No model saved]")
    else:
        print(f"  New CV RMSE: ${training_stats['cv_rmse']:,.2f}")

        if backup_info.get('old_cv_rmse'):
            old_rmse = backup_info['old_cv_rmse']
            new_rmse = training_stats['cv_rmse']
            diff = new_rmse - old_rmse
            direction = "worse" if diff > 0 else "better"
            print(f"  Old CV RMSE: ${old_rmse:,.2f}")
            print(f"  Change: ${abs(diff):,.2f} ({direction})")

        print(f"\n  Model saved: {city_config.model_path}")
        if backup_info.get('model'):
            print(f"  Backup saved: {backup_info['model']}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Rental Data Collection & Model Training Pipeline'
    )
    parser.add_argument('--city', type=str, default='ypsilanti',
                        help='City to process (e.g., ypsilanti, ann_arbor, detroit)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Fetch data but do not train/save model')
    parser.add_argument('--skip-backup', action='store_true',
                        help='Do not backup existing model')
    parser.add_argument('--append', action='store_true',
                        help='Append new listings to existing data instead of replacing')
    parser.add_argument('--verbose', action='store_true',
                        help='Show extra data quality and training logs')
    parser.add_argument('--max-pages', type=int, default=10,
                        help='Maximum number of API pages to fetch (default: 10)')
    parser.add_argument('--list-cities', action='store_true',
                        help='List all available cities and exit')
    args = parser.parse_args()

    # Handle --list-cities
    if args.list_cities:
        print("Available cities:")
        for slug in CityConfig.list_cities():
            config = CityConfig(slug)
            trained = " (trained)" if config.has_trained_model() else ""
            print(f"  {slug}: {config.display_name}, {config.state}{trained}")
        return

    # Load city config
    try:
        city_config = CityConfig(args.city)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("=" * 60)
    print(f"RENTAL DATA COLLECTION & MODEL TRAINING PIPELINE")
    print(f"City: {city_config.display_name}, {city_config.state}")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Backup existing model
    print("\n[Step 1/4] Backing up existing model...")
    backup_info = backup_model(city_config, skip_backup=args.skip_backup)

    # Step 2: Fetch data from RentCast
    print("\n[Step 2/4] Fetching data from RentCast API...")
    df_raw, fetch_metadata = fetch_rentcast_data(city_config, max_pages=args.max_pages, append=args.append)
    fetch_count = len(df_raw)

    # Step 3: Add clusters
    print("\n[Step 3/4] Adding neighborhood clusters...")
    df_clustered, neighborhood_names = add_clusters(df_raw, city_config, verbose=args.verbose)
    cluster_count = len(df_clustered)

    # Step 4: Train and save model
    print("\n[Step 4/4] Training model...")
    training_stats = train_and_save_model(
        df_clustered, city_config,
        neighborhood_names=neighborhood_names,
        dry_run=args.dry_run,
        df_for_cv=df_raw,
        verbose=args.verbose
    )

    # Print summary
    print_summary(city_config, backup_info, fetch_count, cluster_count, training_stats, fetch_metadata)


if __name__ == "__main__":
    main()
