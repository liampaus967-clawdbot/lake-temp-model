"""Tests for lake temperature model."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.features import compute_temporal_features, compute_lake_features


def test_temporal_features_summer():
    """Test temporal features for a summer date."""
    date = datetime(2025, 7, 15)
    features = compute_temporal_features(date)
    
    assert features["month"] == 7
    assert features["is_summer"] == 1
    assert features["is_winter"] == 0
    assert 190 < features["day_of_year"] < 200


def test_temporal_features_winter():
    """Test temporal features for a winter date."""
    date = datetime(2025, 1, 15)
    features = compute_temporal_features(date)
    
    assert features["month"] == 1
    assert features["is_summer"] == 0
    assert features["is_winter"] == 1


def test_lake_features():
    """Test lake characteristic features."""
    features = compute_lake_features(
        surface_area_km2=100,
        max_depth_m=50,
        elevation_m=300,
        latitude=44.5,
    )
    
    assert features["surface_area_km2"] == 100
    assert features["log_surface_area"] == pytest.approx(2.0)
    assert features["latitude"] == 44.5
    assert features["abs_latitude"] == 44.5


def test_cyclical_features():
    """Test that day_of_year sin/cos are properly cyclical."""
    jan1 = compute_temporal_features(datetime(2025, 1, 1))
    dec31 = compute_temporal_features(datetime(2025, 12, 31))
    
    # Should be close (cyclical)
    assert abs(jan1["day_of_year_sin"] - dec31["day_of_year_sin"]) < 0.1
    assert abs(jan1["day_of_year_cos"] - dec31["day_of_year_cos"]) < 0.1
