"""Prediction API for lake temperature model."""

import pandas as pd
from datetime import datetime
from pathlib import Path

from .model import LakeTemperatureModel
from .weather import get_weather_features
from .features import compute_temporal_features, compute_lake_features


def predict_lake_temperature(
    lake_name: str,
    lat: float,
    lon: float,
    date: datetime,
    lake_metadata: dict,
    model_path: str = "models/lake_temp_model.pkl",
) -> dict:
    """
    Predict lake surface temperature for a specific date.
    
    Args:
        lake_name: Name of the lake
        lat: Lake centroid latitude
        lon: Lake centroid longitude
        date: Date to predict for
        lake_metadata: Dict with surface_area_km2, max_depth_m, elevation_m, etc.
        model_path: Path to trained model
        
    Returns:
        Dictionary with prediction and confidence
    """
    # Load model
    model = LakeTemperatureModel(model_path)
    
    # Get weather features
    weather = get_weather_features(lat, lon, date)
    
    # Get temporal features
    temporal = compute_temporal_features(date)
    
    # Get lake features
    lake_feats = compute_lake_features(latitude=lat, **lake_metadata)
    
    # Build feature row
    features = {**weather, **temporal, **lake_feats}
    df = pd.DataFrame([features])
    
    # Predict
    pred = model.predict(df)[0]
    
    return {
        "lake_name": lake_name,
        "date": date.strftime("%Y-%m-%d"),
        "predicted_temp_c": round(pred, 1),
        "predicted_temp_f": round(pred * 9/5 + 32, 1),
        "weather_inputs": {
            "air_temp_c": weather.get("air_temp_2m_c"),
            "solar_radiation": weather.get("solar_radiation_w_m2"),
            "wind_speed": weather.get("wind_speed_10m_m_s"),
        },
    }


# Example lake metadata (would come from a database in production)
LAKE_METADATA = {
    "Lake Champlain": {
        "lat": 44.5,
        "lon": -73.3,
        "surface_area_km2": 1127,
        "max_depth_m": 122,
        "mean_depth_m": 19.5,
        "elevation_m": 29.5,
    },
    "Green River Reservoir": {
        "lat": 44.62,
        "lon": -72.53,
        "surface_area_km2": 2.6,
        "max_depth_m": 15,
        "elevation_m": 430,
    },
}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict lake temperature")
    parser.add_argument("--lake", required=True, help="Lake name")
    parser.add_argument("--date", default=None, help="Date (YYYY-MM-DD), default today")
    parser.add_argument("--model", default="models/lake_temp_model.pkl")
    
    args = parser.parse_args()
    
    date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.now()
    
    if args.lake not in LAKE_METADATA:
        print(f"Unknown lake: {args.lake}")
        print(f"Available: {list(LAKE_METADATA.keys())}")
        exit(1)
    
    meta = LAKE_METADATA[args.lake]
    result = predict_lake_temperature(
        args.lake,
        meta["lat"],
        meta["lon"],
        date,
        meta,
        args.model,
    )
    
    print(f"\n🌡️ {result['lake_name']} - {result['date']}")
    print(f"   Predicted: {result['predicted_temp_c']}°C / {result['predicted_temp_f']}°F")
    print(f"   Air temp:  {result['weather_inputs']['air_temp_c']}°C")
