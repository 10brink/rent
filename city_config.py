"""
city_config.py - City Configuration Management

Provides a CityConfig class for loading and managing city-specific settings
from config/cities.json. Used by pipeline.py, app.py, and rental_price_model.py
to support multi-city model training and predictions.

Usage:
    from city_config import CityConfig

    # Load a specific city
    config = CityConfig('ann_arbor')
    print(config.display_name)  # "Ann Arbor"
    print(config.model_path)    # Path("models/ann_arbor/rental_price_model.cbm")

    # List all available cities
    cities = CityConfig.list_cities()  # ['ypsilanti', 'ann_arbor', ...]

    # Get all configs
    all_configs = CityConfig.get_all_configs()  # {slug: CityConfig, ...}
"""

import json
import os
from pathlib import Path
from typing import Optional


class CityConfig:
    """Configuration for a single city."""

    _config_cache: Optional[dict] = None
    _base_dir: Optional[Path] = None

    def __init__(self, city_slug: str):
        """
        Initialize configuration for a city.

        Args:
            city_slug: Lowercase city identifier (e.g., 'ann_arbor', 'ypsilanti')

        Raises:
            ValueError: If city_slug is not found in config/cities.json
        """
        self.slug = city_slug
        self._load_config()

        if city_slug not in self._config_cache:
            available = list(self._config_cache.keys())
            raise ValueError(
                f"Unknown city '{city_slug}'. Available cities: {available}"
            )

        city_data = self._config_cache[city_slug]
        self.display_name = city_data['display_name']
        self.state = city_data['state']
        self.center_lat = city_data['center_lat']
        self.center_lon = city_data['center_lon']
        self.radius_miles = city_data['radius_miles']
        self.n_clusters = city_data['n_clusters']

    @classmethod
    def _load_config(cls) -> None:
        """Load cities.json config file (cached)."""
        if cls._config_cache is not None:
            return

        # Find base directory (where this file lives)
        cls._base_dir = Path(__file__).parent

        config_path = cls._base_dir / 'config' / 'cities.json'
        if not config_path.exists():
            raise FileNotFoundError(f"City config not found: {config_path}")

        with open(config_path, 'r') as f:
            cls._config_cache = json.load(f)

    @property
    def model_dir(self) -> Path:
        """Directory containing model files for this city."""
        return self._base_dir / 'models' / self.slug

    @property
    def model_path(self) -> Path:
        """Path to the trained CatBoost model file."""
        return self.model_dir / 'rental_price_model.cbm'

    @property
    def metadata_path(self) -> Path:
        """Path to the model metadata JSON file."""
        return self.model_dir / 'rental_price_model_metadata.json'

    @property
    def data_dir(self) -> Path:
        """Directory containing data files for this city."""
        return self._base_dir / 'data' / self.slug

    @property
    def data_path(self) -> Path:
        """Path to the processed rentals CSV (with cluster assignments)."""
        return self.data_dir / 'rentals_with_stats.csv'

    @property
    def raw_data_dir(self) -> Path:
        """Directory for timestamped raw data files."""
        return self.data_dir / 'raw'

    @property
    def backup_dir(self) -> Path:
        """Directory for model backups."""
        return self.model_dir / 'backups'

    def ensure_directories(self) -> None:
        """Create all necessary directories for this city."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def has_trained_model(self) -> bool:
        """Check if a trained model exists for this city."""
        return self.model_path.exists() and self.metadata_path.exists()

    @classmethod
    def list_cities(cls) -> list:
        """Return list of all available city slugs."""
        cls._load_config()
        return list(cls._config_cache.keys())

    @classmethod
    def get_all_configs(cls) -> dict:
        """Return dict mapping city_slug to CityConfig objects."""
        return {slug: cls(slug) for slug in cls.list_cities()}

    @classmethod
    def list_trained_cities(cls) -> list:
        """Return list of city slugs that have trained models."""
        return [slug for slug in cls.list_cities() if cls(slug).has_trained_model()]

    def __repr__(self) -> str:
        return f"CityConfig('{self.slug}')"

    def __str__(self) -> str:
        return f"{self.display_name}, {self.state}"
