"""
Landsat Lake Temperature Raster Pipeline

Automatically fetches, clips, and stores Landsat surface temperature rasters
for Vermont lakes. Designed to run on a schedule to keep rasters up-to-date.
"""

import planetary_computer as pc
from pystac_client import Client
import rioxarray
import geopandas as gpd
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from typing import Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
LANDSAT_COLLECTION = "landsat-c2-l2"
MAX_CLOUD_COVER = 30
THERMAL_BAND = "lwir11"

# Landsat Collection 2 Level-2 scaling for Surface Temperature
# DN * 0.00341802 + 149.0 = Kelvin
SCALE_FACTOR = 0.00341802
OFFSET = 149.0


class LakeTempPipeline:
    """Pipeline for fetching and storing Landsat lake temperature rasters."""
    
    def __init__(
        self,
        lakes_dir: str = "data/lakes",
        rasters_dir: str = "data/rasters",
        metadata_file: str = "data/raster_metadata.json",
    ):
        self.lakes_dir = Path(lakes_dir)
        self.rasters_dir = Path(rasters_dir)
        self.metadata_file = Path(metadata_file)
        
        # Create directories
        self.rasters_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create metadata
        self.metadata = self._load_metadata()
        
        # Initialize STAC catalog
        self.catalog = Client.open(PLANETARY_COMPUTER_URL, modifier=pc.sign_inplace)
    
    def _load_metadata(self) -> dict:
        """Load existing metadata or create empty structure."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {"lakes": {}, "last_run": None}
    
    def _save_metadata(self):
        """Save metadata to file."""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def get_lake_polygons(self) -> dict[str, gpd.GeoDataFrame]:
        """Load all lake polygons from the lakes directory."""
        lakes = {}
        for geojson_file in self.lakes_dir.glob("*.geojson"):
            lake_name = geojson_file.stem.replace("_", " ").title()
            lakes[lake_name] = gpd.read_file(geojson_file)
            logger.info(f"Loaded polygon: {lake_name}")
        return lakes
    
    def search_scenes(
        self,
        bbox: list[float],
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: float = MAX_CLOUD_COVER,
    ) -> list:
        """Search for Landsat scenes covering a bounding box."""
        search = self.catalog.search(
            collections=[LANDSAT_COLLECTION],
            bbox=bbox,
            datetime=f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        )
        return list(search.items())
    
    def process_scene(
        self,
        item,
        lake_polygon: gpd.GeoDataFrame,
        lake_name: str,
        output_dir: Path,
    ) -> Optional[dict]:
        """
        Process a single Landsat scene: clip to lake and convert to temperature.
        
        Returns metadata dict if successful, None if failed.
        """
        scene_date = item.properties.get("datetime")[:10]
        scene_id = item.id
        
        # Create output filename
        safe_lake_name = lake_name.lower().replace(" ", "_")
        output_file = output_dir / f"{safe_lake_name}_{scene_date}.tif"
        
        # Skip if already processed
        if output_file.exists():
            logger.info(f"  Skipping {scene_id} - already processed")
            return None
        
        try:
            # Load thermal band
            lwir = rioxarray.open_rasterio(item.assets[THERMAL_BAND].href)
            
            # Reproject polygon to raster CRS
            lake_proj = lake_polygon.to_crs(lwir.rio.crs)
            
            # Clip to lake boundary
            clipped = lwir.rio.clip(lake_proj.geometry, lake_proj.crs, drop=True)
            
            # Convert to Celsius
            # First to Kelvin using scale/offset, then to Celsius
            temp_kelvin = clipped.astype(np.float32) * SCALE_FACTOR + OFFSET
            temp_celsius = temp_kelvin - 273.15
            
            # Set nodata for invalid temps (outside reasonable range)
            temp_celsius = temp_celsius.where(
                (temp_celsius > -20) & (temp_celsius < 45),
                other=np.nan
            )
            
            # Update attributes
            temp_celsius.attrs["units"] = "celsius"
            temp_celsius.attrs["long_name"] = "Lake Surface Temperature"
            temp_celsius.attrs["source"] = scene_id
            temp_celsius.attrs["date"] = scene_date
            temp_celsius.attrs["lake"] = lake_name
            
            # Set nodata value
            temp_celsius.rio.write_nodata(np.nan, inplace=True)
            
            # Save as Cloud-Optimized GeoTIFF
            temp_celsius.rio.to_raster(
                output_file,
                driver="GTiff",
                compress="LZW",
                tiled=True,
            )
            
            # Calculate stats
            valid_data = temp_celsius.values[~np.isnan(temp_celsius.values)]
            
            if len(valid_data) < 100:
                logger.warning(f"  {scene_id}: Not enough valid pixels ({len(valid_data)})")
                output_file.unlink()  # Remove file
                return None
            
            metadata = {
                "scene_id": scene_id,
                "date": scene_date,
                "cloud_cover": item.properties.get("eo:cloud_cover"),
                "file": str(output_file),
                "temp_min_c": float(np.nanmin(valid_data)),
                "temp_max_c": float(np.nanmax(valid_data)),
                "temp_mean_c": float(np.nanmean(valid_data)),
                "temp_std_c": float(np.nanstd(valid_data)),
                "valid_pixels": int(len(valid_data)),
                "processed_at": datetime.now().isoformat(),
            }
            
            logger.info(
                f"  ✓ {scene_date}: {metadata['temp_mean_c']:.1f}°C "
                f"(range: {metadata['temp_min_c']:.1f} - {metadata['temp_max_c']:.1f})"
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"  ✗ {scene_id}: {str(e)[:50]}")
            return None
    
    def process_lake(
        self,
        lake_name: str,
        lake_polygon: gpd.GeoDataFrame,
        start_date: datetime = None,
        end_date: datetime = None,
        max_scenes: int = 50,
    ) -> list[dict]:
        """Process all available scenes for a single lake."""
        logger.info(f"\nProcessing: {lake_name}")
        
        # Get bounding box
        bounds = lake_polygon.total_bounds
        bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]
        
        # Default date range if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        # Create lake-specific output directory
        safe_lake_name = lake_name.lower().replace(" ", "_")
        output_dir = self.rasters_dir / safe_lake_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Search for scenes
        logger.info(f"  Searching scenes from {start_date.date()} to {end_date.date()}...")
        items = self.search_scenes(bbox, start_date, end_date)
        logger.info(f"  Found {len(items)} scenes")
        
        # Process each scene
        results = []
        for item in items[:max_scenes]:
            result = self.process_scene(item, lake_polygon, lake_name, output_dir)
            if result:
                results.append(result)
        
        # Update metadata
        if lake_name not in self.metadata["lakes"]:
            self.metadata["lakes"][lake_name] = {"scenes": []}
        
        # Merge with existing scenes (avoid duplicates)
        existing_dates = {s["date"] for s in self.metadata["lakes"][lake_name]["scenes"]}
        for result in results:
            if result["date"] not in existing_dates:
                self.metadata["lakes"][lake_name]["scenes"].append(result)
        
        # Sort by date
        self.metadata["lakes"][lake_name]["scenes"].sort(key=lambda x: x["date"], reverse=True)
        
        return results
    
    def run(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        days_back: int = 365,
        max_scenes_per_lake: int = 50,
    ):
        """Run the pipeline for all lakes."""
        # Handle date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=days_back)
        
        logger.info("=" * 60)
        logger.info("Lake Temperature Raster Pipeline")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info("=" * 60)
        
        # Load lake polygons
        lakes = self.get_lake_polygons()
        
        if not lakes:
            logger.error("No lake polygons found in {self.lakes_dir}")
            return
        
        logger.info(f"Found {len(lakes)} lakes")
        
        # Process each lake
        total_processed = 0
        for lake_name, lake_polygon in lakes.items():
            results = self.process_lake(
                lake_name, lake_polygon,
                start_date=start_date, end_date=end_date,
                max_scenes=max_scenes_per_lake
            )
            total_processed += len(results)
        
        # Save metadata
        self.metadata["last_run"] = datetime.now().isoformat()
        self._save_metadata()
        
        logger.info("=" * 60)
        logger.info(f"Pipeline complete! Processed {total_processed} new scenes")
        logger.info(f"Metadata saved to: {self.metadata_file}")
    
    def get_latest_raster(self, lake_name: str) -> Optional[Path]:
        """Get the path to the most recent raster for a lake."""
        if lake_name not in self.metadata["lakes"]:
            return None
        
        scenes = self.metadata["lakes"][lake_name]["scenes"]
        if not scenes:
            return None
        
        # Already sorted by date descending
        return Path(scenes[0]["file"])
    
    def list_available_rasters(self, lake_name: str) -> list[dict]:
        """List all available rasters for a lake."""
        if lake_name not in self.metadata["lakes"]:
            return []
        return self.metadata["lakes"][lake_name]["scenes"]


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lake Temperature Raster Pipeline")
    parser.add_argument("--lakes-dir", default="data/lakes", help="Directory with lake polygons")
    parser.add_argument("--rasters-dir", default="data/rasters", help="Output directory for rasters")
    parser.add_argument("--days", type=int, default=365, help="Days of history to fetch (from end-date)")
    parser.add_argument("--max-scenes", type=int, default=50, help="Max scenes per lake")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD), default: today")
    
    args = parser.parse_args()
    
    # Parse dates
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=args.days)
    
    pipeline = LakeTempPipeline(
        lakes_dir=args.lakes_dir,
        rasters_dir=args.rasters_dir,
    )
    pipeline.run(start_date=start_date, end_date=end_date, max_scenes_per_lake=args.max_scenes)


if __name__ == "__main__":
    main()
