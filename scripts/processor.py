# scripts/processor.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import argparse
from pathlib import Path

class ExoplanetPCAProcessor:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.feature_names = [
            'period', 'duration', 'depth', 'stellar_radius', 
            'stellar_temp', 'impact_parameter'
        ]
        self.fitted = False
        
    def load_data(self, filepath):
        """Load exoplanet data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, df):
        """Clean and preprocess the data"""
        # Map common column names from NASA datasets
        column_mapping = {
            'pl_orbper': 'period',
            'pl_trandur': 'duration', 
            'pl_trandep': 'depth',
            'st_rad': 'stellar_radius',
            'st_teff': 'stellar_temp',
            'pl_imppar': 'impact_parameter',
            'pl_rade': 'planet_radius',
            'disposition': 'classification',
            'koi_disposition': 'classification'
        }
        
        # Rename columns if they exist
        df_clean = df.rename(columns=column_mapping)
        
        # Select relevant features
        feature_cols = [col for col in self.feature_names if col in df_clean.columns]
        
        if len(feature_cols) < 3:
            raise ValueError("Insufficient feature columns found in dataset")
        
        # Extract features and handle missing values
        X = df_clean[feature_cols].copy()
        
        # Fill missing values with median
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col].fillna(X[col].median(), inplace=True)
        
        # Log transform skewed features
        log_features = ['period', 'depth']
        for feature in log_features:
            if feature in X.columns:
                X[feature] = np.log10(X[feature] + 1e-10)
        
        # Extract labels if available
        y = None
        if 'classification' in df_clean.columns:
            y = df_clean['classification'].map({
                'CONFIRMED': 2,
                'CANDIDATE': 1, 
                'FALSE POSITIVE': 0,
                'Confirmed': 2,
                'Candidate': 1,
                'False Positive': 0
            }).fillna(0)
        
        return X.values, y, feature_cols
    
    def fit_transform(self, X, save_path='models/'):
        """Fit PCA and transform data"""
        Path(save_path).mkdir(exist_ok=True)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        X_pca = self.pca.fit_transform(X_scaled)
        
        self.fitted = True
        
        # Save models
        joblib.dump(self.scaler, f'{save_path}/pca_scaler.joblib')
        joblib.dump(self.pca, f'{save_path}/pca_model.joblib')
        
        # Save PCA statistics
        pca_stats = {
            'n_components': self.n_components,
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_).tolist(),
            'feature_names': self.feature_names,
            'components': self.pca.components_.tolist()
        }
        
        with open(f'{save_path}/pca_stats.json', 'w') as f:
            json.dump(pca_stats, f, indent=2)
        
        print(f"PCA Model saved. Explained variance ratio: {self.pca.explained_variance_ratio_}")
        print(f"Cumulative variance explained: {np.cumsum(self.pca.explained_variance_ratio_)}")
        
        return X_pca
    
    def transform_new_data(self, X):
        """Transform new data using fitted PCA"""
        if not self.fitted:
            raise ValueError("PCA model not fitted. Call fit_transform first.")
        
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def export_for_js(self, output_path='data/'):
        """Export PCA parameters for JavaScript usage"""
        Path(output_path).mkdir(exist_ok=True)
        
        if not self.fitted:
            raise ValueError("PCA model not fitted")
        
        js_export = {
            'pca_components': self.pca.components_.tolist(),
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist(),
            'n_components': self.n_components,
            'feature_names': self.feature_names
        }
        
        with open(f'{output_path}/pca_params.json', 'w') as f:
            json.dump(js_export, f, indent=2)
        
        print(f"PCA parameters exported to {output_path}/pca_params.json")

def main():
    parser = argparse.ArgumentParser(description='Process exoplanet data with PCA')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', default='data/', help='Output directory')
    parser.add_argument('--components', '-c', type=int, default=3, help='Number of PCA components')
    parser.add_argument('--test-size', '-t', type=float, default=0.2, help='Test set size')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = ExoplanetPCAProcessor(n_components=args.components)
    
    # Load and preprocess data
    df = processor.load_data(args.input)
    if df is None:
        return
    
    X, y, feature_names = processor.preprocess_data(df)
    print(f"Preprocessed data shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Split data
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
    else:
        X_train, X_test = train_test_split(X, test_size=args.test_size, random_state=42)
        y_train = y_test = None
    
    # Fit PCA
    X_train_pca = processor.fit_transform(X_train)
    X_test_pca = processor.transform_new_data(X_test)
    
    # Save processed data
    Path(args.output).mkdir(exist_ok=True)
    
    np.save(f'{args.output}/X_train_pca.npy', X_train_pca)
    np.save(f'{args.output}/X_test_pca.npy', X_test_pca)
    
    if y_train is not None:
        np.save(f'{args.output}/y_train.npy', y_train)
        np.save(f'{args.output}/y_test.npy', y_test)
    
    # Export for JavaScript
    processor.export_for_js(args.output)
    
    print(f"Processing complete. Output saved to {args.output}")
    print(f"PCA Components shape: {X_train_pca.shape}")

if __name__ == "__main__":
    main()