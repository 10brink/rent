"""
rental_price_model.py - Model Training, Evaluation, and Prediction Utilities

The machine learning core. Trains and compares multiple regression models,
with CatBoost as the winner. Also provides utilities for loading saved models
and making predictions.

Key functions:
  - main(): Train and compare all models, optionally save the best one
  - load_saved_model(): Load a trained model from disk
  - predict_price(): Make a prediction for a single property
  - geocode_address(): Convert address to lat/long via Nominatim
  - find_nearest_cluster(): Map coordinates to nearest neighborhood cluster

Usage:
  python3 rental_price_model.py --remove-outliers --save  # Train and save

As a library:
  from rental_price_model import load_saved_model, predict_price
  model, metadata = load_saved_model()
  price = predict_price(model, beds=2, baths=1, sqft=900, property_type='Apartment', cluster_id=0)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def clean_outliers(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Remove rows with impossible/erroneous values before imputation.

    Rules:
    - SqFt < 200: Too small to be livable
    - SqFt > 5000: Likely typo (e.g., 5850 sqft 3-bed is probably 585)
    """
    before = len(df)

    # Remove impossibly small sqft (< 200)
    df = df[~(df['SqFt'] < 200)]

    # Remove impossibly large sqft (> 5000, likely typos)
    df = df[~(df['SqFt'] > 5000)]

    removed = before - len(df)
    if verbose:
        pct = (removed / before * 100) if before else 0.0
        print(f"Outlier cleanup: removed {removed} rows ({pct:.1f}%), remaining {len(df)}")

    return df


def compute_city_medians(df: pd.DataFrame) -> dict:
    """
    Compute SqFt medians per bed count from city's own data.

    Returns dict mapping bed count to median sqft.
    Used for city-specific imputation instead of hardcoded values.
    """
    # Convert SqFt to numeric if needed
    sqft_series = pd.to_numeric(df['SqFt'], errors='coerce')
    beds_series = pd.to_numeric(df['Beds'], errors='coerce')

    medians = {}
    for beds in range(6):
        subset = sqft_series[beds_series == beds].dropna()
        if len(subset) > 0:
            medians[beds] = int(subset.median())
        else:
            # Fallback to overall median or default
            medians[beds] = 870
    return medians


def impute_missing_values(df: pd.DataFrame, city_medians: dict = None, verbose: bool = False) -> pd.DataFrame:
    """
    Impute missing Beds, Baths, and SqFt values.

    Uses city-specific medians if provided, otherwise falls back to defaults.

    Strategy:
    1. Impute Beds from SqFt (using 450 sqft as studio/1-bed boundary)
    2. Impute Baths from Beds (1 bath default, 2 for 4+ beds)
    3. Impute SqFt from Beds (using city-specific medians or defaults)
    4. Fallback to Property Type medians for remaining missing values
    """
    df = df.copy()

    # Use city-specific medians or fall back to defaults (Ypsilanti values)
    sqft_by_beds = city_medians if city_medians else {0: 388, 1: 629, 2: 870, 3: 1167, 4: 1568, 5: 1700}

    # Track imputation counts
    beds_imputed = 0
    baths_imputed = 0
    sqft_imputed = 0

    # 1. Impute Beds when missing but SqFt known (450 sqft = studio/1-bed boundary)
    mask = df['Beds'].isna() & df['SqFt'].notna()
    beds_imputed = mask.sum()
    df.loc[mask & (df['SqFt'] < 450), 'Beds'] = 0       # Studio
    df.loc[mask & (df['SqFt'] >= 450) & (df['SqFt'] < 750), 'Beds'] = 1
    df.loc[mask & (df['SqFt'] >= 750) & (df['SqFt'] < 1100), 'Beds'] = 2
    df.loc[mask & (df['SqFt'] >= 1100) & (df['SqFt'] < 1500), 'Beds'] = 3
    df.loc[mask & (df['SqFt'] >= 1500), 'Beds'] = 4

    # 2. Impute Baths when missing but Beds known
    mask = df['Baths'].isna() & df['Beds'].notna()
    baths_imputed = mask.sum()
    df.loc[mask & (df['Beds'] < 4), 'Baths'] = 1.0
    df.loc[mask & (df['Beds'] >= 4), 'Baths'] = 2.0

    # 3. Impute SqFt when missing but Beds known (using city-specific medians)
    for beds, sqft in sqft_by_beds.items():
        mask = df['SqFt'].isna() & (df['Beds'] == beds)
        sqft_imputed += mask.sum()
        df.loc[mask, 'SqFt'] = sqft
    # Handle 5+ beds
    mask = df['SqFt'].isna() & (df['Beds'] > 5)
    sqft_imputed += mask.sum()
    df.loc[mask, 'SqFt'] = sqft_by_beds.get(5, 1700)

    # 4. Impute remaining missing Beds/Baths by Property Type
    type_defaults = {
        'Condo': (2, 1.0),
        'Apartment': (2, 1.0),
        'Single Family': (3, 1.5),
        'Multi-Family': (2, 1.0),
        'Townhouse': (2, 1.5),
    }
    for ptype, (beds, baths) in type_defaults.items():
        mask = (df['Property Type'] == ptype)
        beds_before = df['Beds'].isna().sum()
        df.loc[mask & df['Beds'].isna(), 'Beds'] = beds
        beds_imputed += beds_before - df['Beds'].isna().sum()

        baths_before = df['Baths'].isna().sum()
        df.loc[mask & df['Baths'].isna(), 'Baths'] = baths
        baths_imputed += baths_before - df['Baths'].isna().sum()

    # 5. Final fallback for any remaining SqFt (use overall median)
    remaining_sqft = df['SqFt'].isna().sum()
    if remaining_sqft > 0:
        sqft_imputed += remaining_sqft
        df.loc[df['SqFt'].isna(), 'SqFt'] = 870  # Overall median

    # 6. Final fallback for any remaining Beds/Baths
    remaining_beds = df['Beds'].isna().sum()
    if remaining_beds > 0:
        beds_imputed += remaining_beds
        df.loc[df['Beds'].isna(), 'Beds'] = 2  # Default

    remaining_baths = df['Baths'].isna().sum()
    if remaining_baths > 0:
        baths_imputed += remaining_baths
        df.loc[df['Baths'].isna(), 'Baths'] = 1.0  # Default

    if verbose:
        print(f"Imputed values - Beds: {beds_imputed}, Baths: {baths_imputed}, SqFt: {sqft_imputed}")

    return df


def remove_outliers_iqr(X: pd.DataFrame, y: pd.Series, multiplier: float = 1.5) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Remove price outliers using IQR method.

    Args:
        X: Feature DataFrame
        y: Price Series
        multiplier: IQR multiplier (default 1.5 for standard outliers)

    Returns:
        Filtered X, filtered y, and stats dict
    """
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    mask = (y >= lower_bound) & (y <= upper_bound)

    n_removed = (~mask).sum()
    removed_prices = y[~mask].sort_values()

    stats = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'n_original': len(y),
        'n_removed': n_removed,
        'n_remaining': mask.sum(),
        'removed_low': removed_prices[removed_prices < lower_bound].tolist(),
        'removed_high': removed_prices[removed_prices > upper_bound].tolist(),
    }

    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), stats


def load_and_prepare_data(filepath: str, remove_outliers: bool = False, iqr_multiplier: float = 1.5) -> tuple:
    """Load CSV and prepare features and target.

    Args:
        filepath: Path to CSV file
        remove_outliers: Whether to remove price outliers using IQR method
        iqr_multiplier: IQR multiplier for outlier detection (default 1.5)

    Returns:
        X (features), y (target), outlier_stats (or None if not removing outliers)
    """
    df = pd.read_csv(filepath)

    # Defensive cleanup: normalize missing values and coerce numeric fields
    df = df.replace({"N/A": np.nan, "": np.nan})
    for col in ["Price", "Beds", "Baths", "SqFt", "Latitude", "Longitude", "Year Built"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Price is required for training
    df = df.dropna(subset=["Price"])

    # Data is already cleaned in pipeline.py add_clusters() - no need to clean again

    # Define features (excluding Lat/Long - location is captured by Cluster_ID)
    feature_cols = ['Beds', 'Baths', 'SqFt', 'Property Type', 'Cluster_ID']

    # Filter to only include columns that exist
    available_cols = [col for col in feature_cols if col in df.columns]

    X = df[available_cols].copy()
    y = df['Price'].copy()

    # Convert Cluster_ID to string for categorical handling
    if 'Cluster_ID' in X.columns:
        X['Cluster_ID'] = X['Cluster_ID'].astype(str)

    # Optionally remove outliers
    outlier_stats = None
    if remove_outliers:
        X, y, outlier_stats = remove_outliers_iqr(X, y, multiplier=iqr_multiplier)

    return X, y, outlier_stats


def train_model(X_train: pd.DataFrame, y_train: pd.Series, cat_features: list) -> CatBoostRegressor:
    """Train CatBoost regression model."""
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,
        random_state=42,
        verbose=100
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: CatBoostRegressor, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> dict:
    """Calculate and print evaluation metrics."""
    predictions = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    print(f"\n{dataset_name} Metrics:")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  R²:   {r2:.4f}")

    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def run_cross_validation(X: pd.DataFrame, y: pd.Series, cat_features: list) -> None:
    """Perform 5-fold cross-validation."""
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,
        random_state=42,
        verbose=0  # Quiet during CV
    )

    # cross_val_score uses negative MSE, so we negate and sqrt for RMSE
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)

    print("\n5-Fold Cross-Validation Results:")
    print(f"  RMSE scores: {[f'${x:,.2f}' for x in cv_rmse]}")
    print(f"  Mean RMSE:   ${cv_rmse.mean():,.2f}")
    print(f"  Std RMSE:    ${cv_rmse.std():,.2f}")


def plot_feature_importance(model: CatBoostRegressor, feature_names: list, output_path: str) -> None:
    """Extract and plot feature importances."""
    importances = model.get_feature_importance()

    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    print("\nFeature Importance Ranking:")
    for i, (name, imp) in enumerate(zip(sorted_names, sorted_importances), 1):
        print(f"  {i}. {name}: {imp:.2f}")

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_names)), sorted_importances[::-1], align='center')
    plt.yticks(range(len(sorted_names)), sorted_names[::-1])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Rental Price Prediction - Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nFeature importance plot saved to: {output_path}")


def create_sklearn_pipeline(model, numeric_features: list, categorical_features: list) -> Pipeline:
    """Create a pipeline with preprocessing for sklearn models."""
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])


def compare_models(X_train, X_test, y_train, y_test, X, y, cat_features, numeric_features):
    """Train and compare all models."""
    results = []

    # 1. Linear Regression
    print("\n" + "="*50)
    print("LINEAR REGRESSION")
    print("="*50)
    lr_pipeline = create_sklearn_pipeline(LinearRegression(), numeric_features, cat_features)
    lr_pipeline.fit(X_train, y_train)
    lr_train_pred = lr_pipeline.predict(X_train)
    lr_test_pred = lr_pipeline.predict(X_test)
    lr_cv = cross_val_score(lr_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    lr_cv_rmse = np.sqrt(-lr_cv)
    print(f"Train RMSE: ${np.sqrt(mean_squared_error(y_train, lr_train_pred)):,.2f}")
    print(f"Test RMSE:  ${np.sqrt(mean_squared_error(y_test, lr_test_pred)):,.2f}")
    print(f"Test R²:    {r2_score(y_test, lr_test_pred):.4f}")
    print(f"CV RMSE:    ${lr_cv_rmse.mean():,.2f} (±${lr_cv_rmse.std():,.2f})")
    results.append(('Linear Regression', np.sqrt(mean_squared_error(y_test, lr_test_pred)), lr_cv_rmse.mean(), lr_cv_rmse.std()))

    # 2. Random Forest
    print("\n" + "="*50)
    print("RANDOM FOREST")
    print("="*50)
    rf_pipeline = create_sklearn_pipeline(
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        numeric_features, cat_features
    )
    rf_pipeline.fit(X_train, y_train)
    rf_train_pred = rf_pipeline.predict(X_train)
    rf_test_pred = rf_pipeline.predict(X_test)
    rf_cv = cross_val_score(rf_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    rf_cv_rmse = np.sqrt(-rf_cv)
    print(f"Train RMSE: ${np.sqrt(mean_squared_error(y_train, rf_train_pred)):,.2f}")
    print(f"Test RMSE:  ${np.sqrt(mean_squared_error(y_test, rf_test_pred)):,.2f}")
    print(f"Test R²:    {r2_score(y_test, rf_test_pred):.4f}")
    print(f"CV RMSE:    ${rf_cv_rmse.mean():,.2f} (±${rf_cv_rmse.std():,.2f})")
    results.append(('Random Forest', np.sqrt(mean_squared_error(y_test, rf_test_pred)), rf_cv_rmse.mean(), rf_cv_rmse.std()))

    # 3. XGBoost
    print("\n" + "="*50)
    print("XGBOOST")
    print("="*50)
    xgb_pipeline = create_sklearn_pipeline(
        XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0),
        numeric_features, cat_features
    )
    xgb_pipeline.fit(X_train, y_train)
    xgb_train_pred = xgb_pipeline.predict(X_train)
    xgb_test_pred = xgb_pipeline.predict(X_test)
    xgb_cv = cross_val_score(xgb_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    xgb_cv_rmse = np.sqrt(-xgb_cv)
    print(f"Train RMSE: ${np.sqrt(mean_squared_error(y_train, xgb_train_pred)):,.2f}")
    print(f"Test RMSE:  ${np.sqrt(mean_squared_error(y_test, xgb_test_pred)):,.2f}")
    print(f"Test R²:    {r2_score(y_test, xgb_test_pred):.4f}")
    print(f"CV RMSE:    ${xgb_cv_rmse.mean():,.2f} (±${xgb_cv_rmse.std():,.2f})")
    results.append(('XGBoost', np.sqrt(mean_squared_error(y_test, xgb_test_pred)), xgb_cv_rmse.mean(), xgb_cv_rmse.std()))

    # 4. CatBoost
    print("\n" + "="*50)
    print("CATBOOST")
    print("="*50)
    cat_model = CatBoostRegressor(
        iterations=500, learning_rate=0.1, depth=6,
        cat_features=cat_features, random_state=42, verbose=0
    )
    cat_model.fit(X_train, y_train)
    cat_train_pred = cat_model.predict(X_train)
    cat_test_pred = cat_model.predict(X_test)
    cat_cv = cross_val_score(
        CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, cat_features=cat_features, random_state=42, verbose=0),
        X, y, cv=5, scoring='neg_mean_squared_error'
    )
    cat_cv_rmse = np.sqrt(-cat_cv)
    print(f"Train RMSE: ${np.sqrt(mean_squared_error(y_train, cat_train_pred)):,.2f}")
    print(f"Test RMSE:  ${np.sqrt(mean_squared_error(y_test, cat_test_pred)):,.2f}")
    print(f"Test R²:    {r2_score(y_test, cat_test_pred):.4f}")
    print(f"CV RMSE:    ${cat_cv_rmse.mean():,.2f} (±${cat_cv_rmse.std():,.2f})")
    results.append(('CatBoost', np.sqrt(mean_squared_error(y_test, cat_test_pred)), cat_cv_rmse.mean(), cat_cv_rmse.std()))

    return results, cat_model


def plot_comparison(results: list, output_path: str) -> None:
    """Plot model comparison."""
    models = [r[0] for r in results]
    test_rmse = [r[1] for r in results]
    cv_rmse = [r[2] for r in results]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, test_rmse, width, label='Test RMSE')
    bars2 = ax.bar(x + width/2, cv_rmse, width, label='CV RMSE')

    ax.set_ylabel('RMSE ($)')
    ax.set_title('Model Comparison - Rental Price Prediction')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'${height:,.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'${height:,.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nModel comparison plot saved to: {output_path}")


def compute_cluster_centroids(filepath: str) -> dict:
    """Compute cluster centroids from the training data."""
    df = pd.read_csv(filepath)
    centroids = {}
    for cid in df['Cluster_ID'].unique():
        subset = df[df['Cluster_ID'] == cid]
        centroids[int(cid)] = {
            'latitude': float(subset['Latitude'].mean()),
            'longitude': float(subset['Longitude'].mean()),
            'count': int(len(subset)),
        }
    return centroids


def find_nearest_cluster(lat: float, lon: float, centroids: dict) -> int:
    """Find the nearest cluster based on Euclidean distance to centroids."""
    min_dist = float('inf')
    nearest_cluster = 0
    for cid, centroid in centroids.items():
        dist = (lat - centroid['latitude'])**2 + (lon - centroid['longitude'])**2
        if dist < min_dist:
            min_dist = dist
            nearest_cluster = cid
    return nearest_cluster


def normalize_address(address: str) -> list:
    """
    Generate address variations to improve geocoding success.
    Returns a list of addresses to try, starting with the original.
    """
    import re

    variations = [address]

    def add_variation(value: str) -> None:
        cleaned = re.sub(r'\s+', ' ', value).strip()
        if cleaned and cleaned not in variations:
            variations.append(cleaned)

    # Common abbreviation mappings
    abbreviations = {
        r'\bSt\b\.?': 'Street',
        r'\bRd\b\.?': 'Road',
        r'\bAve\b\.?': 'Avenue',
        r'\bBlvd\b\.?': 'Boulevard',
        r'\bDr\b\.?': 'Drive',
        r'\bLn\b\.?': 'Lane',
        r'\bCt\b\.?': 'Court',
        r'\bPl\b\.?': 'Place',
        r'\bCir\b\.?': 'Circle',
        r'\bPkwy\b\.?': 'Parkway',
        r'\bHwy\b\.?': 'Highway',
        r'\bApt\b\.?\s*#?\d*': '',
        r'\bUnit\b\.?\s*#?\d*': '',
        r'\b#\d+': '',
    }

    # State abbreviations
    state_abbrevs = {
        r'\bMI\b': 'Michigan',
        r'\bOH\b': 'Ohio',
        r'\bIN\b': 'Indiana',
    }

    # Try merging the last two words of a street name (e.g., "May Apple" -> "Mayapple")
    stripped = re.sub(r'[.,]', '', address).strip()
    tokens = re.split(r'\s+', stripped)
    suffixes = {
        'st', 'street', 'rd', 'road', 'ave', 'avenue', 'blvd', 'boulevard',
        'dr', 'drive', 'ln', 'lane', 'ct', 'court', 'pl', 'place',
        'cir', 'circle', 'pkwy', 'parkway', 'hwy', 'highway'
    }
    for idx, token in enumerate(tokens):
        if token.lower() in suffixes and idx >= 3:
            merged = tokens[:]
            merged[idx - 2:idx] = [''.join(tokens[idx - 2:idx])]
            add_variation(' '.join(merged))
            break

    # Try expanding street abbreviations
    expanded = address
    for abbrev, full in abbreviations.items():
        expanded = re.sub(abbrev, full, expanded, flags=re.IGNORECASE)
    if expanded != address:
        variations.append(expanded)

    # Try expanding state abbreviations
    state_expanded = address
    for abbrev, full in state_abbrevs.items():
        state_expanded = re.sub(abbrev, full, state_expanded, flags=re.IGNORECASE)
    if state_expanded != address:
        variations.append(state_expanded)

    # Try both expansions
    both_expanded = expanded
    for abbrev, full in state_abbrevs.items():
        both_expanded = re.sub(abbrev, full, both_expanded, flags=re.IGNORECASE)
    if both_expanded not in variations:
        variations.append(both_expanded)

    # Try removing zip code
    no_zip = re.sub(r'\b\d{5}(-\d{4})?\b', '', address).strip().rstrip(',')
    if no_zip != address and no_zip not in variations:
        variations.append(no_zip)

    # Try inserting commas between street, city, and state if missing
    comma_pattern = r'^\s*(.+?)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\s+(MI|Michigan|OH|Ohio|IN|Indiana)\s*$'
    for addr in list(variations):
        if ',' in addr:
            continue
        match = re.match(comma_pattern, addr, flags=re.IGNORECASE)
        if match:
            street, city, state = match.groups()
            add_variation(f"{street}, {city}, {state}")
        with_state_comma = re.sub(r'\s+(MI|Michigan|OH|Ohio|IN|Indiana)\b', r', \1', addr, flags=re.IGNORECASE)
        if with_state_comma != addr:
            add_variation(with_state_comma)

    # Try removing punctuation
    no_punct = re.sub(r'[.,]', '', address)
    add_variation(no_punct)

    # Try removing street suffix entirely (last resort)
    no_suffix = re.sub(r'\b(St|Street|Rd|Road|Ave|Avenue|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court|Pl|Place|Cir|Circle)\b\.?', '', address, flags=re.IGNORECASE)
    no_suffix = re.sub(r'\s+', ' ', no_suffix).strip()
    if no_suffix not in variations:
        variations.append(no_suffix)

    return variations


def geocode_address(address: str) -> tuple:
    """
    Convert an address to lat/long coordinates using Nominatim (OpenStreetMap).
    Tries multiple address variations to improve success rate.

    Returns (latitude, longitude) or raises ValueError if not found.
    """
    import urllib.request
    import urllib.parse

    variations = normalize_address(address)

    for addr in variations:
        encoded_address = urllib.parse.quote(addr)
        url = f"https://nominatim.openstreetmap.org/search?q={encoded_address}&format=json&limit=1"

        req = urllib.request.Request(url, headers={'User-Agent': 'RentalPricePredictor/1.0'})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
        except Exception:
            continue

    raise ValueError(f"Could not geocode address: {address}")


def save_model(model: CatBoostRegressor, feature_names: list, cat_features: list,
               output_path: str = 'rental_price_model.cbm',
               cluster_centroids: dict = None,
               cv_rmse: float = None) -> None:
    """Save CatBoost model and metadata for later use."""
    # Save the model
    model.save_model(output_path)

    # Save metadata as JSON
    metadata = {
        'feature_names': feature_names,
        'categorical_features': cat_features,
        'model_type': 'CatBoostRegressor',
        'model_file': output_path,
        'cluster_centroids': cluster_centroids,
        'cv_rmse': cv_rmse,
    }
    metadata_path = output_path.replace('.cbm', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")


def load_saved_model(model_path: str = None, city: str = None) -> tuple:
    """
    Load a saved CatBoost model and its metadata.

    Args:
        model_path: Direct path to model file (takes precedence if provided)
        city: City slug to load model for (uses city config paths)

    Returns:
        tuple of (model, metadata)
    """
    if model_path is None and city is None:
        # Default to legacy path for backwards compatibility
        model_path = 'rental_price_model.cbm'

    if city is not None:
        from city_config import CityConfig
        config = CityConfig(city)
        model_path = str(config.model_path)
        metadata_path = str(config.metadata_path)
    else:
        metadata_path = model_path.replace('.cbm', '_metadata.json')

    model = CatBoostRegressor()
    model.load_model(model_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return model, metadata


def predict_price(model: CatBoostRegressor, beds: int, baths: float, sqft: int,
                  property_type: str, cluster_id: int = None,
                  lat: float = None, lon: float = None,
                  address: str = None, metadata: dict = None,
                  extra_features: dict = None) -> float:
    """
    Make a single prediction using the loaded model.

    Provide cluster_id directly, OR provide (lat, lon), OR provide address.
    If using lat/lon or address, metadata with cluster_centroids is required.
    """
    if cluster_id is None:
        if metadata is None or 'cluster_centroids' not in metadata:
            raise ValueError("metadata with cluster_centroids required when not providing cluster_id")

        centroids = metadata['cluster_centroids']
        # Convert string keys back to int (JSON serialization converts int keys to strings)
        centroids = {int(k): v for k, v in centroids.items()}

        if address is not None:
            lat, lon = geocode_address(address)
            print(f"Geocoded '{address}' to ({lat:.4f}, {lon:.4f})")

        if lat is None or lon is None:
            raise ValueError("Must provide cluster_id, (lat, lon), or address")

        cluster_id = find_nearest_cluster(lat, lon, centroids)
        print(f"Assigned to Cluster {cluster_id}")

    row = {
        'Beds': beds,
        'Baths': baths,
        'SqFt': sqft,
        'Property Type': property_type,
        'Cluster_ID': str(cluster_id)
    }
    if extra_features:
        row.update(extra_features)

    feature_names = metadata.get('feature_names') if metadata else None
    if feature_names:
        missing = [name for name in feature_names if name not in row]
        if missing:
            raise ValueError(f"Missing features for prediction: {missing}")
        X = pd.DataFrame([[row[name] for name in feature_names]], columns=feature_names)
    else:
        X = pd.DataFrame([row])
    pred = model.predict(X)[0]
    if metadata and metadata.get('target_transform') == 'log1p':
        pred = np.expm1(pred)
        pred = max(pred, 0)
    return pred


def print_outlier_stats(stats: dict) -> None:
    """Print outlier removal statistics."""
    print("\n" + "="*50)
    print("OUTLIER REMOVAL (IQR Method)")
    print("="*50)
    print(f"Q1 (25th percentile): ${stats['Q1']:,.0f}")
    print(f"Q3 (75th percentile): ${stats['Q3']:,.0f}")
    print(f"IQR: ${stats['IQR']:,.0f}")
    print(f"Lower bound: ${stats['lower_bound']:,.0f}")
    print(f"Upper bound: ${stats['upper_bound']:,.0f}")
    print(f"\nOriginal samples: {stats['n_original']}")
    print(f"Removed: {stats['n_removed']} ({100*stats['n_removed']/stats['n_original']:.1f}%)")
    print(f"Remaining: {stats['n_remaining']}")

    if stats['removed_low']:
        print(f"\nRemoved low outliers: {[f'${p:,.0f}' for p in stats['removed_low']]}")
    if stats['removed_high']:
        print(f"Removed high outliers: {[f'${p:,.0f}' for p in stats['removed_high']]}")


def main(remove_outliers: bool = False, iqr_multiplier: float = 1.5, save: bool = False):
    # Load data
    print("Loading data...")
    X, y, outlier_stats = load_and_prepare_data(
        'rentals_with_stats.csv',
        remove_outliers=remove_outliers,
        iqr_multiplier=iqr_multiplier
    )

    if outlier_stats:
        print_outlier_stats(outlier_stats)

    print(f"\nDataset: {len(X)} samples")
    print(f"Features: {list(X.columns)}")
    print(f"Target (Price): ${y.min():,.0f} - ${y.max():,.0f} (mean: ${y.mean():,.0f})")

    # Identify categorical and numeric features
    cat_features = ['Property Type', 'Cluster_ID']
    cat_features = [f for f in cat_features if f in X.columns]
    numeric_features = [f for f in X.columns if f not in cat_features]

    print(f"Categorical features: {cat_features}")
    print(f"Numeric features: {numeric_features}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Compare all models
    results, catboost_model = compare_models(X_train, X_test, y_train, y_test, X, y, cat_features, numeric_features)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Test RMSE':>12} {'CV RMSE':>12} {'CV Std':>12}")
    print("-"*58)
    for name, test_rmse, cv_rmse, cv_std in sorted(results, key=lambda x: x[2]):
        print(f"{name:<20} ${test_rmse:>10,.2f} ${cv_rmse:>10,.2f} ${cv_std:>10,.2f}")

    # Plot comparison
    plot_comparison(results, 'model_comparison.png')

    # Feature importance (CatBoost)
    plot_feature_importance(catboost_model, list(X.columns), 'feature_importance.png')

    # Save model if requested
    if save:
        centroids = compute_cluster_centroids('rentals_with_stats.csv')
        # Get CV RMSE for CatBoost from results
        catboost_cv_rmse = next(r[2] for r in results if r[0] == 'CatBoost')
        save_model(catboost_model, list(X.columns), cat_features,
                   cluster_centroids=centroids, cv_rmse=catboost_cv_rmse)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Rental Price Prediction Model Comparison')
    parser.add_argument('--remove-outliers', action='store_true',
                        help='Remove price outliers using IQR method')
    parser.add_argument('--iqr-multiplier', type=float, default=1.5,
                        help='IQR multiplier for outlier detection (default: 1.5)')
    parser.add_argument('--save', action='store_true',
                        help='Save the trained CatBoost model to rental_price_model.cbm')
    args = parser.parse_args()

    main(remove_outliers=args.remove_outliers, iqr_multiplier=args.iqr_multiplier, save=args.save)
