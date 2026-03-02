"""Fetch HRRR weather data for model features."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Note: herbie requires eccodes/cfgrib which can be tricky to install
# Alternative: use AWS/GCS direct access or pre-processed data


def get_weather_features(
    lat: float,
    lon: float,
    date: datetime,
    source: str = "hrrr",
) -> dict:
    """
    Get weather features for a location and date.
    
    Args:
        lat: Latitude
        lon: Longitude
        date: Date to fetch weather for
        source: Weather data source ('hrrr' or 'era5')
        
    Returns:
        Dictionary of weather features
    """
    try:
        from herbie import Herbie
        
        # Get HRRR analysis for the date (use 12Z for midday)
        H = Herbie(date, model="hrrr", product="sfc", fxx=0)
        
        # Variables we need
        variables = {
            "TMP:2 m": "air_temp_2m_k",
            "DSWRF:surface": "solar_radiation_w_m2",
            "UGRD:10 m": "wind_u_10m_m_s",
            "VGRD:10 m": "wind_v_10m_m_s",
            "RH:2 m": "relative_humidity_pct",
        }
        
        features = {}
        for var, name in variables.items():
            try:
                ds = H.xarray(var)
                # Extract value at lat/lon
                val = ds.sel(latitude=lat, longitude=lon, method="nearest")
                features[name] = float(val.values)
            except:
                features[name] = None
        
        # Calculate wind speed from components
        if features.get("wind_u_10m_m_s") and features.get("wind_v_10m_m_s"):
            features["wind_speed_10m_m_s"] = np.sqrt(
                features["wind_u_10m_m_s"]**2 + features["wind_v_10m_m_s"]**2
            )
        
        # Convert air temp to Celsius
        if features.get("air_temp_2m_k"):
            features["air_temp_2m_c"] = features["air_temp_2m_k"] - 273.15
        
        return features
        
    except ImportError:
        print("Warning: herbie not available, using fallback weather data")
        return get_weather_features_fallback(lat, lon, date)


def get_weather_features_fallback(lat: float, lon: float, date: datetime) -> dict:
    """
    Fallback weather features using Open-Meteo API (free, no API key).
    """
    import requests
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date.strftime("%Y-%m-%d"),
        "end_date": date.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation",
        "timezone": "UTC",
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if "hourly" not in data:
        return {}
    
    hourly = data["hourly"]
    
    # Use midday values (index 12 for 12:00 UTC)
    idx = 12 if len(hourly["time"]) > 12 else 0
    
    return {
        "air_temp_2m_c": hourly["temperature_2m"][idx],
        "relative_humidity_pct": hourly["relative_humidity_2m"][idx],
        "wind_speed_10m_m_s": hourly["wind_speed_10m"][idx],
        "solar_radiation_w_m2": hourly["shortwave_radiation"][idx],
    }


def get_weather_for_dates(
    lat: float,
    lon: float,
    dates: list[datetime],
) -> pd.DataFrame:
    """
    Get weather features for multiple dates.
    
    Args:
        lat: Latitude
        lon: Longitude
        dates: List of dates
        
    Returns:
        DataFrame with weather features
    """
    records = []
    for date in dates:
        features = get_weather_features(lat, lon, date)
        features["date"] = date.strftime("%Y-%m-%d")
        features["lat"] = lat
        features["lon"] = lon
        records.append(features)
    
    return pd.DataFrame(records)


if __name__ == "__main__":
    # Test with Lake Champlain center
    lat, lon = 44.5, -73.3
    date = datetime(2025, 7, 4)
    
    print(f"Weather for {lat}, {lon} on {date.date()}:")
    features = get_weather_features(lat, lon, date)
    for k, v in features.items():
        print(f"  {k}: {v}")
