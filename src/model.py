"""Lake temperature prediction model."""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from .features import FEATURE_COLUMNS, TARGET_COLUMN


class LakeTemperatureModel:
    """XGBoost model for predicting lake surface temperatures."""
    
    def __init__(self, model_path: Path = None):
        self.model = None
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.model_path = model_path
        self.metadata = {}
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def train(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        """
        Train the model on a feature matrix.
        
        Args:
            df: Feature matrix from build_feature_matrix()
            target_col: Target column name
            test_size: Fraction for test set
            random_state: Random seed
            
        Returns:
            Dictionary with training metrics
        """
        # Filter to available features
        available_features = [c for c in self.feature_columns if c in df.columns]
        print(f"Using {len(available_features)} features: {available_features}")
        
        # Drop rows with missing target
        df = df.dropna(subset=[target_col])
        
        X = df[available_features]
        y = df[target_col]
        
        # Handle missing values in features
        X = X.fillna(X.median())
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train XGBoost
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_r2": r2_score(y_test, y_pred_test),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "features_used": available_features,
        }
        
        print(f"\n📊 Model Performance:")
        print(f"   Train RMSE: {metrics['train_rmse']:.2f}°C")
        print(f"   Test RMSE:  {metrics['test_rmse']:.2f}°C")
        print(f"   Test MAE:   {metrics['test_mae']:.2f}°C")
        print(f"   Test R²:    {metrics['test_r2']:.3f}")
        
        # Feature importance
        importance = dict(zip(available_features, self.model.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        metrics["feature_importance"] = importance
        
        print(f"\n🎯 Top Features:")
        for feat, imp in list(importance.items())[:5]:
            print(f"   {feat}: {imp:.3f}")
        
        self.metadata = {
            "trained_at": datetime.now().isoformat(),
            "metrics": metrics,
            "feature_columns": available_features,
        }
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict lake temperatures.
        
        Args:
            df: Feature matrix
            
        Returns:
            Array of predicted temperatures (°C)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        available_features = [c for c in self.feature_columns if c in df.columns]
        X = df[available_features].fillna(df[available_features].median())
        
        return self.model.predict(X)
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_columns": self.feature_columns,
                "metadata": self.metadata,
            }, f)
        print(f"✅ Model saved to {path}")
    
    def load(self, path: Path):
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.metadata = data["metadata"]
        print(f"✅ Model loaded from {path}")


def train_model(data_path: str, output_path: str = "models/lake_temp_model.pkl"):
    """Train model from processed data file."""
    df = pd.read_parquet(data_path)
    
    model = LakeTemperatureModel()
    metrics = model.train(df)
    model.save(output_path)
    
    return model, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "predict"])
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--model", default="models/lake_temp_model.pkl")
    
    args = parser.parse_args()
    
    if args.action == "train":
        train_model(args.data, args.model)
