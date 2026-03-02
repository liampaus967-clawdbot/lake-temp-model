# Lake Temperature Model

A machine learning model to predict lake surface water temperatures using satellite imagery and weather data.

## Overview

This model predicts lake surface temperatures by combining:
- **Landsat thermal imagery** (training/validation data)
- **HRRR weather data** (air temp, solar radiation, wind)
- **Lake characteristics** (surface area, depth, elevation)

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Landsat LST    │     │   HRRR Weather  │     │ Lake Metadata   │
│  (Ground Truth) │     │   (Features)    │     │  (Features)     │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Temperature Model     │
                    │   (XGBoost / LightGBM)  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Predicted Lake Temp    │
                    │  (Daily, Any Location)  │
                    └─────────────────────────┘
```

## Features

### Input Features
- Air temperature (2m) from HRRR
- Solar radiation (downward shortwave)
- Wind speed (10m)
- Relative humidity
- Day of year (seasonality)
- Lake surface area
- Lake max depth
- Lake elevation
- Latitude

### Target
- Lake surface temperature (°C) from Landsat thermal band

## Data Sources

| Source | Resolution | Frequency | Use |
|--------|------------|-----------|-----|
| Landsat 8/9 Collection 2 | 100m | 8-16 days | Training labels |
| HRRR | 3km | Hourly | Weather features |
| NHD Plus | - | Static | Lake polygons/metadata |

## Project Structure

```
lake-temp-model/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # Raw Landsat/weather downloads
│   ├── processed/        # Cleaned training data
│   └── lakes/            # Lake polygons and metadata
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── landsat.py        # Landsat data fetching
│   ├── weather.py        # HRRR data fetching
│   ├── features.py       # Feature engineering
│   ├── model.py          # Model training/inference
│   └── predict.py        # Prediction API
├── models/
│   └── .gitkeep          # Trained model artifacts
└── tests/
    └── test_model.py
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch training data for a lake
python -m src.landsat --lake "Lake Champlain" --start 2023-01-01 --end 2025-12-31

# Train model
python -m src.model train --data data/processed/training.parquet

# Predict today's temperature
python -m src.predict --lake "Lake Champlain" --date 2026-03-02
```

## License

MIT
