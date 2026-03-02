"""Fetch Landsat surface temperature data from Microsoft Planetary Computer."""

import planetary_computer as pc
from pystac_client import Client
import rioxarray
import geopandas as gpd
import numpy as np
from pyproj import Transformer
from datetime import datetime
from pathlib import Path
import json


PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def get_catalog():
    """Get authenticated Planetary Computer STAC catalog."""
    return Client.open(PLANETARY_COMPUTER_URL, modifier=pc.sign_inplace)


def search_landsat_scenes(
    bbox: list[float],
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 30.0,
) -> list:
    """
    Search for Landsat Collection 2 Level-2 scenes.
    
    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_cloud_cover: Maximum cloud cover percentage
        
    Returns:
        List of STAC items
    """
    catalog = get_catalog()
    
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    
    return list(search.items())


def fetch_surface_temperature(
    item,
    geometry: gpd.GeoDataFrame,
    output_path: Path = None,
) -> dict:
    """
    Fetch and clip Landsat surface temperature to a lake polygon.
    
    Args:
        item: STAC item from search
        geometry: GeoDataFrame with lake polygon
        output_path: Optional path to save clipped raster
        
    Returns:
        Dictionary with temperature statistics
    """
    # Load thermal band
    lwir = rioxarray.open_rasterio(item.assets['lwir11'].href)
    raster_crs = lwir.rio.crs
    
    # Reproject geometry to match raster
    geom_utm = geometry.to_crs(raster_crs)
    
    # Clip to lake boundary
    clipped = lwir.rio.clip(geom_utm.geometry, geom_utm.crs, drop=True)
    
    # Convert DN to Celsius
    # Landsat Collection 2 Level-2: Scale=0.00341802, Offset=149.0 (Kelvin)
    temp_kelvin = clipped.values.astype(float) * 0.00341802 + 149.0
    temp_celsius = temp_kelvin - 273.15
    
    # Filter valid water temperatures
    valid = temp_celsius[(temp_celsius > 0) & (temp_celsius < 40)]
    
    result = {
        "scene_id": item.id,
        "date": item.properties.get("datetime")[:10],
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "temp_min_c": float(np.min(valid)) if len(valid) > 0 else None,
        "temp_max_c": float(np.max(valid)) if len(valid) > 0 else None,
        "temp_mean_c": float(np.mean(valid)) if len(valid) > 0 else None,
        "temp_std_c": float(np.std(valid)) if len(valid) > 0 else None,
        "valid_pixels": len(valid),
    }
    
    # Save if requested
    if output_path:
        clipped.rio.to_raster(output_path)
        result["raster_path"] = str(output_path)
    
    return result


def build_training_dataset(
    lake_polygon: gpd.GeoDataFrame,
    lake_name: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    max_cloud_cover: float = 25.0,
) -> list[dict]:
    """
    Build training dataset by fetching all available Landsat scenes for a lake.
    
    Args:
        lake_polygon: GeoDataFrame with lake boundary
        lake_name: Name of the lake (for logging)
        start_date: Start date
        end_date: End date
        output_dir: Directory to save outputs
        max_cloud_cover: Maximum cloud cover
        
    Returns:
        List of temperature records
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get bounding box
    bounds = lake_polygon.total_bounds
    bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]
    
    print(f"Searching Landsat scenes for {lake_name}...")
    items = search_landsat_scenes(bbox, start_date, end_date, max_cloud_cover)
    print(f"Found {len(items)} scenes")
    
    records = []
    for i, item in enumerate(items):
        print(f"Processing {i+1}/{len(items)}: {item.id}")
        try:
            record = fetch_surface_temperature(item, lake_polygon)
            record["lake_name"] = lake_name
            records.append(record)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save records
    output_file = output_dir / f"{lake_name.lower().replace(' ', '_')}_landsat.json"
    with open(output_file, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records to {output_file}")
    
    return records


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Landsat data for a lake")
    parser.add_argument("--lake", required=True, help="Lake name")
    parser.add_argument("--polygon", required=True, help="Path to lake polygon GeoJSON")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    
    args = parser.parse_args()
    
    lake_poly = gpd.read_file(args.polygon)
    build_training_dataset(lake_poly, args.lake, args.start, args.end, args.output)
