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

# Visible bands for clarity estimation
BLUE_BAND = "blue"    # B2 (482nm)
GREEN_BAND = "green"  # B3 (562nm)
RED_BAND = "red"      # B4 (655nm)

# Landsat Collection 2 Level-2 scaling for Surface Temperature
# DN * 0.00341802 + 149.0 = Kelvin
SCALE_FACTOR = 0.00341802
OFFSET = 149.0

# Landsat Collection 2 Level-2 scaling for Surface Reflectance
# DN * 0.0000275 - 0.2 = reflectance
SR_SCALE = 0.0000275
SR_OFFSET = -0.2


def calculate_secchi_depth(blue: np.ndarray, green: np.ndarray) -> np.ndarray:
    """
    Estimate Secchi disk depth (water clarity) from blue/green band ratio.
    
    Uses empirical algorithm based on Olmanson et al. (2008) and others.
    Secchi Depth (m) = exp(a + b * ln(Blue/Green))
    
    Coefficients are approximate - would need local calibration for accuracy.
    """
    # Avoid division by zero
    green_safe = np.where(green > 0, green, np.nan)
    ratio = blue / green_safe
    
    # Empirical coefficients (approximate for temperate lakes)
    # These would ideally be calibrated with in-situ measurements
    a = 1.5  # intercept
    b = 2.5  # slope
    
    # Calculate Secchi depth in meters
    secchi = np.exp(a + b * np.log(ratio))
    
    # Clip to reasonable range (0.1 to 15 meters)
    secchi = np.clip(secchi, 0.1, 15.0)
    
    return secchi


def calculate_turbidity(red: np.ndarray) -> np.ndarray:
    """
    Estimate turbidity (NTU) from red band reflectance.
    
    Higher red reflectance = more suspended particles = higher turbidity.
    Uses simplified linear relationship.
    """
    # Empirical relationship (approximate)
    # Turbidity (NTU) ≈ red_reflectance * scale
    turbidity = red * 500  # Approximate scaling
    
    # Clip to reasonable range
    turbidity = np.clip(turbidity, 0, 100)
    
    return turbidity


class LakeTempPipeline:
    """Pipeline for fetching and storing Landsat lake temperature rasters."""
    
    def __init__(
        self,
        lakes_dir: str = "data/lakes",
        rasters_dir: str = "data/rasters",
        metadata_file: str = "data/raster_metadata.json",
        include_clarity: bool = False,
        save_context: bool = False,
        context_buffer_km: float = 5.0,
    ):
        self.lakes_dir = Path(lakes_dir)
        self.rasters_dir = Path(rasters_dir)
        self.metadata_file = Path(metadata_file)
        self.include_clarity = include_clarity
        self.save_context = save_context
        self.context_buffer_km = context_buffer_km
        
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
        Optionally also generates water clarity raster.
        
        Returns metadata dict if successful, None if failed.
        """
        scene_date = item.properties.get("datetime")[:10]
        scene_id = item.id
        
        # Create output filename
        safe_lake_name = lake_name.lower().replace(" ", "_")
        output_file = output_dir / f"{safe_lake_name}_{scene_date}.tif"
        clarity_file = output_dir / f"{safe_lake_name}_{scene_date}_clarity.tif"
        
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
            
            # Process clarity if enabled
            if self.include_clarity:
                try:
                    clarity_meta = self._process_clarity(
                        item, lake_proj, scene_date, scene_id, clarity_file
                    )
                    if clarity_meta:
                        metadata.update(clarity_meta)
                except Exception as e:
                    logger.warning(f"  Clarity processing failed: {str(e)[:50]}")
            
            # Save context raster (buffered area around lake) if enabled
            if self.save_context:
                try:
                    context_meta = self._save_context_raster(
                        item, lake_polygon, scene_date, scene_id, output_dir, safe_lake_name
                    )
                    if context_meta:
                        metadata.update(context_meta)
                except Exception as e:
                    logger.warning(f"  Context raster failed: {str(e)[:50]}")
            
            logger.info(
                f"  ✓ {scene_date}: {metadata['temp_mean_c']:.1f}°C "
                f"(range: {metadata['temp_min_c']:.1f} - {metadata['temp_max_c']:.1f})"
                + (f" | Secchi: {metadata.get('secchi_mean_m', 'N/A')}m" if self.include_clarity and 'secchi_mean_m' in metadata else "")
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"  ✗ {scene_id}: {str(e)[:50]}")
            return None
    
    def _process_clarity(
        self,
        item,
        lake_proj: gpd.GeoDataFrame,
        scene_date: str,
        scene_id: str,
        output_file: Path,
    ) -> Optional[dict]:
        """
        Process water clarity (Secchi depth) from visible bands.
        
        Returns metadata dict with clarity stats.
        """
        # Load blue and green bands
        blue = rioxarray.open_rasterio(item.assets[BLUE_BAND].href)
        green = rioxarray.open_rasterio(item.assets[GREEN_BAND].href)
        
        # Clip to lake
        blue_clipped = blue.rio.clip(lake_proj.geometry, lake_proj.crs, drop=True)
        green_clipped = green.rio.clip(lake_proj.geometry, lake_proj.crs, drop=True)
        
        # Convert to surface reflectance
        blue_sr = blue_clipped.astype(np.float32) * SR_SCALE + SR_OFFSET
        green_sr = green_clipped.astype(np.float32) * SR_SCALE + SR_OFFSET
        
        # Mask negative reflectance (invalid/cloud shadow)
        blue_sr = blue_sr.where(blue_sr > 0, other=np.nan)
        green_sr = green_sr.where(green_sr > 0, other=np.nan)
        
        # Calculate Secchi depth
        secchi = calculate_secchi_depth(blue_sr.values, green_sr.values)
        
        # Create xarray with same coordinates as blue band
        secchi_da = blue_clipped.copy(data=secchi)
        secchi_da.attrs["units"] = "meters"
        secchi_da.attrs["long_name"] = "Estimated Secchi Disk Depth"
        secchi_da.attrs["source"] = scene_id
        secchi_da.attrs["date"] = scene_date
        secchi_da.rio.write_nodata(np.nan, inplace=True)
        
        # Save clarity raster
        secchi_da.rio.to_raster(
            output_file,
            driver="GTiff",
            compress="LZW",
            tiled=True,
        )
        
        # Calculate stats
        valid_secchi = secchi[~np.isnan(secchi)]
        
        if len(valid_secchi) < 100:
            return None
        
        return {
            "clarity_file": str(output_file),
            "secchi_min_m": float(np.nanmin(valid_secchi)),
            "secchi_max_m": float(np.nanmax(valid_secchi)),
            "secchi_mean_m": round(float(np.nanmean(valid_secchi)), 2),
            "secchi_std_m": round(float(np.nanstd(valid_secchi)), 2),
        }
    
    def _save_context_raster(
        self,
        item,
        lake_polygon: gpd.GeoDataFrame,
        scene_date: str,
        scene_id: str,
        output_dir: Path,
        safe_lake_name: str,
    ) -> Optional[dict]:
        """
        Save a buffered context raster around the lake (includes surrounding area).
        
        Returns metadata dict with context file path.
        """
        context_file = output_dir / f"{safe_lake_name}_{scene_date}_context.tif"
        
        # Skip if already exists
        if context_file.exists():
            return {"context_file": str(context_file)}
        
        # Load thermal band
        lwir = rioxarray.open_rasterio(item.assets[THERMAL_BAND].href)
        
        # Get lake bounds and add buffer (convert km to degrees, ~0.01 deg ≈ 1km)
        buffer_deg = self.context_buffer_km * 0.01
        bounds = lake_polygon.total_bounds  # [minx, miny, maxx, maxy]
        buffered_bounds = [
            bounds[0] - buffer_deg,
            bounds[1] - buffer_deg,
            bounds[2] + buffer_deg,
            bounds[3] + buffer_deg,
        ]
        
        # Reproject bounds to raster CRS
        from shapely.geometry import box
        from pyproj import Transformer
        
        # Create transformer from WGS84 to raster CRS
        transformer = Transformer.from_crs("EPSG:4326", lwir.rio.crs, always_xy=True)
        
        # Transform the buffered bounds
        min_x, min_y = transformer.transform(buffered_bounds[0], buffered_bounds[1])
        max_x, max_y = transformer.transform(buffered_bounds[2], buffered_bounds[3])
        
        # Clip to buffered bounding box
        try:
            context_clip = lwir.rio.clip_box(min_x, min_y, max_x, max_y)
        except Exception:
            # If clip_box fails, try with original bounds
            context_clip = lwir
        
        # Convert to Celsius
        temp_kelvin = context_clip.astype(np.float32) * SCALE_FACTOR + OFFSET
        temp_celsius = temp_kelvin - 273.15
        
        # Set nodata for invalid temps
        temp_celsius = temp_celsius.where(
            (temp_celsius > -20) & (temp_celsius < 45),
            other=np.nan
        )
        
        # Update attributes
        temp_celsius.attrs["units"] = "celsius"
        temp_celsius.attrs["long_name"] = "Lake Surface Temperature (Context)"
        temp_celsius.attrs["source"] = scene_id
        temp_celsius.attrs["date"] = scene_date
        temp_celsius.rio.write_nodata(np.nan, inplace=True)
        
        # Save as GeoTIFF
        temp_celsius.rio.to_raster(
            context_file,
            driver="GTiff",
            compress="LZW",
            tiled=True,
        )
        
        return {"context_file": str(context_file)}
    
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
    parser.add_argument("--include-clarity", action="store_true", 
                        help="Also generate water clarity (Secchi depth) rasters")
    parser.add_argument("--save-context", action="store_true",
                        help="Save buffered context rasters (lake + surrounding area)")
    parser.add_argument("--context-buffer", type=float, default=5.0,
                        help="Buffer distance in km for context rasters (default: 5)")
    
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
        include_clarity=args.include_clarity,
        save_context=args.save_context,
        context_buffer_km=args.context_buffer,
    )
    pipeline.run(start_date=start_date, end_date=end_date, max_scenes_per_lake=args.max_scenes)


if __name__ == "__main__":
    main()
