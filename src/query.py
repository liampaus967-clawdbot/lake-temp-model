"""
Query latest lake temperature rasters.

Provides functions to get the most recent cloud-free raster for each lake.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional


def load_metadata(metadata_file: str = "data/raster_metadata.json") -> dict:
    """Load raster metadata."""
    with open(metadata_file) as f:
        return json.load(f)


def get_latest_rasters(metadata_file: str = "data/raster_metadata.json") -> list[dict]:
    """
    Get the most recent raster for each lake.
    
    Returns list of dicts with lake info and latest raster details.
    """
    metadata = load_metadata(metadata_file)
    
    results = []
    for lake_name, lake_data in metadata.get("lakes", {}).items():
        scenes = lake_data.get("scenes", [])
        if not scenes:
            continue
        
        # Scenes are sorted by date descending, so first is latest
        latest = scenes[0]
        
        results.append({
            "lake_name": lake_name,
            "date": latest["date"],
            "days_ago": (datetime.now() - datetime.strptime(latest["date"], "%Y-%m-%d")).days,
            "temp_mean_c": round(latest["temp_mean_c"], 1),
            "temp_mean_f": round(latest["temp_mean_c"] * 9/5 + 32, 1),
            "temp_min_c": round(latest["temp_min_c"], 1),
            "temp_max_c": round(latest["temp_max_c"], 1),
            "cloud_cover": latest["cloud_cover"],
            "file": latest["file"],
            "valid_pixels": latest["valid_pixels"],
        })
    
    # Sort by date (most recent first)
    results.sort(key=lambda x: x["date"], reverse=True)
    return results


def get_lake_raster(
    lake_name: str,
    metadata_file: str = "data/raster_metadata.json",
    max_age_days: Optional[int] = None,
) -> Optional[dict]:
    """
    Get the latest raster for a specific lake.
    
    Args:
        lake_name: Name of the lake (case-insensitive)
        metadata_file: Path to metadata file
        max_age_days: Optional max age filter
        
    Returns:
        Dict with raster info, or None if not found
    """
    metadata = load_metadata(metadata_file)
    
    # Find lake (case-insensitive)
    lake_key = None
    for key in metadata.get("lakes", {}).keys():
        if key.lower() == lake_name.lower():
            lake_key = key
            break
    
    if not lake_key:
        return None
    
    scenes = metadata["lakes"][lake_key].get("scenes", [])
    if not scenes:
        return None
    
    latest = scenes[0]
    days_ago = (datetime.now() - datetime.strptime(latest["date"], "%Y-%m-%d")).days
    
    if max_age_days and days_ago > max_age_days:
        return None
    
    return {
        "lake_name": lake_key,
        "date": latest["date"],
        "days_ago": days_ago,
        "temp_mean_c": round(latest["temp_mean_c"], 1),
        "temp_mean_f": round(latest["temp_mean_c"] * 9/5 + 32, 1),
        "temp_min_c": round(latest["temp_min_c"], 1),
        "temp_max_c": round(latest["temp_max_c"], 1),
        "cloud_cover": latest["cloud_cover"],
        "file": latest["file"],
        "valid_pixels": latest["valid_pixels"],
    }


def get_all_lakes_summary(metadata_file: str = "data/raster_metadata.json") -> dict:
    """
    Get summary of all lakes with latest data.
    
    Returns dict ready for API response.
    """
    latest = get_latest_rasters(metadata_file)
    
    return {
        "generated_at": datetime.now().isoformat(),
        "lake_count": len(latest),
        "lakes": latest,
    }


def print_summary(metadata_file: str = "data/raster_metadata.json"):
    """Print a formatted summary of latest rasters."""
    latest = get_latest_rasters(metadata_file)
    
    print("\n🌡️  VERMONT LAKE TEMPERATURES - Latest Available")
    print("=" * 65)
    print(f"{'Lake':<25} {'Date':<12} {'Temp (°F)':<10} {'Age':<8}")
    print("-" * 65)
    
    for lake in latest:
        print(
            f"{lake['lake_name']:<25} "
            f"{lake['date']:<12} "
            f"{lake['temp_mean_f']:>6.1f}°F   "
            f"{lake['days_ago']:>3}d ago"
        )
    
    print("=" * 65)
    print(f"Data from Landsat 8/9 thermal band (100m resolution)")
    print()


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Query lake temperature rasters")
    parser.add_argument("--lake", help="Get data for specific lake")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--metadata", default="data/raster_metadata.json")
    
    args = parser.parse_args()
    
    if args.lake:
        result = get_lake_raster(args.lake, args.metadata)
        if result:
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"\n{result['lake_name']}")
                print(f"  Date: {result['date']} ({result['days_ago']} days ago)")
                print(f"  Temp: {result['temp_mean_f']}°F ({result['temp_mean_c']}°C)")
                print(f"  Range: {result['temp_min_c']}°C - {result['temp_max_c']}°C")
                print(f"  File: {result['file']}")
        else:
            print(f"No data found for '{args.lake}'")
            sys.exit(1)
    else:
        if args.json:
            print(json.dumps(get_all_lakes_summary(args.metadata), indent=2))
        else:
            print_summary(args.metadata)
