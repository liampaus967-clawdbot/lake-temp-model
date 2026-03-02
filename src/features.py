"""Feature engineering for lake temperature prediction."""

import numpy as np
import pandas as pd
from datetime import datetime


def compute_temporal_features(date: datetime) -> dict:
    """
    Compute temporal features from a date.
    
    Args:
        date: The date
        
    Returns:
        Dictionary of temporal features
    """
    day_of_year = date.timetuple().tm_yday
    
    return {
        "day_of_year": day_of_year,
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365),
        "month": date.month,
        "is_summer": 1 if date.month in [6, 7, 8] else 0,
        "is_winter": 1 if date.month in [12, 1, 2] else 0,
    }


def compute_lake_features(
    surface_area_km2: float,
    max_depth_m: float = None,
    mean_depth_m: float = None,
    elevation_m: float = None,
    latitude: float = None,
) -> dict:
    """
    Compute lake characteristic features.
    
    Args:
        surface_area_km2: Lake surface area in km²
        max_depth_m: Maximum depth in meters
        mean_depth_m: Mean depth in meters
        elevation_m: Lake elevation in meters
        latitude: Lake centroid latitude
        
    Returns:
        Dictionary of lake features
    """
    features = {
        "surface_area_km2": surface_area_km2,
        "log_surface_area": np.log10(surface_area_km2) if surface_area_km2 else None,
    }
    
    if max_depth_m:
        features["max_depth_m"] = max_depth_m
        features["log_max_depth"] = np.log10(max_depth_m)
    
    if mean_depth_m:
        features["mean_depth_m"] = mean_depth_m
    
    if elevation_m:
        features["elevation_m"] = elevation_m
    
    if latitude:
        features["latitude"] = latitude
        features["abs_latitude"] = abs(latitude)
    
    # Volume proxy (if we have area and depth)
    if surface_area_km2 and mean_depth_m:
        features["volume_proxy_km3"] = surface_area_km2 * mean_depth_m / 1000
    
    return features


def compute_lag_features(
    df: pd.DataFrame,
    temp_col: str = "temp_mean_c",
    date_col: str = "date",
    lags: list[int] = [1, 7, 14, 30],
) -> pd.DataFrame:
    """
    Compute lagged temperature features.
    
    Args:
        df: DataFrame with temperature data
        temp_col: Column name for temperature
        date_col: Column name for date
        lags: List of lag days
        
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    for lag in lags:
        df[f"temp_lag_{lag}d"] = df[temp_col].shift(lag)
    
    # Rolling means
    df["temp_rolling_7d"] = df[temp_col].rolling(7, min_periods=1).mean()
    df["temp_rolling_30d"] = df[temp_col].rolling(30, min_periods=1).mean()
    
    return df


def build_feature_matrix(
    landsat_data: pd.DataFrame,
    weather_data: pd.DataFrame,
    lake_metadata: dict,
) -> pd.DataFrame:
    """
    Build complete feature matrix for training.
    
    Args:
        landsat_data: DataFrame with Landsat temperature observations
        weather_data: DataFrame with weather features
        lake_metadata: Dictionary with lake characteristics
        
    Returns:
        Feature matrix ready for training
    """
    # Merge Landsat and weather on date
    df = landsat_data.merge(weather_data, on="date", how="inner")
    
    # Add temporal features
    temporal = df["date"].apply(
        lambda x: pd.Series(compute_temporal_features(pd.to_datetime(x)))
    )
    df = pd.concat([df, temporal], axis=1)
    
    # Add lake features
    lake_feats = compute_lake_features(**lake_metadata)
    for k, v in lake_feats.items():
        df[k] = v
    
    # Add lag features
    df = compute_lag_features(df)
    
    return df


# Feature columns for the model
FEATURE_COLUMNS = [
    # Weather
    "air_temp_2m_c",
    "solar_radiation_w_m2",
    "wind_speed_10m_m_s",
    "relative_humidity_pct",
    # Temporal
    "day_of_year_sin",
    "day_of_year_cos",
    "is_summer",
    "is_winter",
    # Lake characteristics
    "log_surface_area",
    "latitude",
    "elevation_m",
    # Lag features (when available)
    "temp_lag_1d",
    "temp_rolling_7d",
]

TARGET_COLUMN = "temp_mean_c"
