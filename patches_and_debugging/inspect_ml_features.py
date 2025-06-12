import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class MLFeatureInspector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        
    def load_model(self):
        """Load the trained model"""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✅ Loaded model from {self.model_path}")
        print(f"   Model type: {type(self.model).__name__}")
        
    def set_feature_names(self, feature_names):
        """Set feature names if they weren't stored with the model"""
        self.feature_names = feature_names
        
    def inspect_model_structure(self):
        """Inspect the model structure and parameters"""
        print("\n" + "="*70)
        print("MODEL STRUCTURE INSPECTION")
        print("="*70)
        
        # Check model type
        model_type = type(self.model).__name__
        
        if hasattr(self.model, 'n_features_in_'):
            print(f"\n📊 Number of input features: {self.model.n_features_in_}")
        
        if model_type == 'RandomForestRegressor':
            self._inspect_random_forest()
        elif model_type == 'GradientBoostingRegressor':
            self._inspect_gradient_boosting()
        elif model_type == 'LinearRegression':
            self._inspect_linear_regression()
        elif hasattr(self.model, 'estimators_'):
            # Ensemble methods
            print(f"\n🌳 Ensemble model with {len(self.model.estimators_)} estimators")
        
        # General model parameters
        if hasattr(self.model, 'get_params'):
            params = self.model.get_params()
            print("\n🔧 Model Parameters:")
            for key, value in sorted(params.items()):
                if not key.endswith('_') and value is not None:
                    print(f"   {key}: {value}")
    
    def _inspect_random_forest(self):
        """Inspect Random Forest specific features"""
        print("\n🌲 Random Forest Model Details:")
        print(f"   Number of trees: {self.model.n_estimators}")
        print(f"   Max depth: {self.model.max_depth}")
        print(f"   Min samples split: {self.model.min_samples_split}")
        print(f"   Min samples leaf: {self.model.min_samples_leaf}")
        
        if hasattr(self.model, 'feature_importances_'):
            self._show_feature_importance()
    
    def _inspect_gradient_boosting(self):
        """Inspect Gradient Boosting specific features"""
        print("\n🚀 Gradient Boosting Model Details:")
        print(f"   Number of estimators: {self.model.n_estimators}")
        print(f"   Learning rate: {self.model.learning_rate}")
        print(f"   Max depth: {self.model.max_depth}")
        print(f"   Subsample: {self.model.subsample}")
        
        if hasattr(self.model, 'feature_importances_'):
            self._show_feature_importance()
    
    def _inspect_linear_regression(self):
        """Inspect Linear Regression specific features"""
        print("\n📈 Linear Regression Model Details:")
        if hasattr(self.model, 'coef_'):
            print(f"   Coefficients shape: {self.model.coef_.shape}")
            print(f"   Intercept: {self.model.intercept_:.4f}")
            self._show_linear_coefficients()
    
    def _show_feature_importance(self):
        """Display feature importance for tree-based models"""
        if not hasattr(self.model, 'feature_importances_'):
            print("   ⚠️  No feature importances available")
            return
        
        importances = self.model.feature_importances_
        n_features = len(importances)
        
        print(f"\n📊 Feature Importances (Top 10):")
        
        # Create feature names if not provided
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        for i in range(min(10, n_features)):
            idx = indices[i]
            print(f"   {i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        # Plot feature importance
        self._plot_feature_importance(importances, indices)
    
    def _show_linear_coefficients(self):
        """Display coefficients for linear models"""
        if not hasattr(self.model, 'coef_'):
            return
        
        coefs = self.model.coef_
        n_features = len(coefs)
        
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        print(f"\n📊 Model Coefficients:")
        
        # Sort by absolute value
        abs_coefs = np.abs(coefs)
        indices = np.argsort(abs_coefs)[::-1]
        
        for i in range(min(10, n_features)):
            idx = indices[i]
            print(f"   {self.feature_names[idx]}: {coefs[idx]:.4f}")
    
    def _plot_feature_importance(self, importances, indices):
        """Create feature importance plot"""
        plt.figure(figsize=(10, 6))
        
        # Take top 15 features
        top_n = min(15, len(importances))
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_names = [self.feature_names[i] for i in top_indices]
        
        # Create horizontal bar plot
        y_pos = np.arange(top_n)
        plt.barh(y_pos, top_importances, align='center', alpha=0.8)
        plt.yticks(y_pos, top_names)
        plt.xlabel('Feature Importance')
        plt.title('Top Feature Importances in ML Model')
        plt.tight_layout()
        
        # Save plot
        output_path = Path("output") / "ml_feature_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Feature importance plot saved to: {output_path}")
        plt.close()
    
    def create_feature_documentation(self):
        """Create documentation for the features"""
        if self.feature_names is None:
            print("\n⚠️  No feature names provided. Using generic names.")
            return
        
        doc_content = f"""# ML Model Feature Documentation
Generated from: {self.model_path}
Model Type: {type(self.model).__name__}
Number of Features: {len(self.feature_names) if self.feature_names else 'Unknown'}

## Feature List:
"""
        
        for i, name in enumerate(self.feature_names):
            doc_content += f"{i+1}. {name}\n"
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            doc_content += "\n## Feature Importances:\n"
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i, idx in enumerate(indices):
                doc_content += f"{i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}\n"
        
        # Save documentation
        doc_path = Path("output") / "ml_model_features.md"
        with open(doc_path, 'w') as f:
            f.write(doc_content)
        
        print(f"\n📄 Feature documentation saved to: {doc_path}")

# Example feature names based on common soccer prediction features
# You should replace these with your actual feature names
EXAMPLE_FEATURE_NAMES = [
    "home_avg_goals_for",
    "home_avg_goals_against", 
    "home_win_rate",
    "home_recent_form",
    "home_goals_at_home",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "away_win_rate", 
    "away_recent_form",
    "away_goals_at_away",
    "head_to_head_goal_diff",
    "home_days_since_last_game",
    "away_days_since_last_game",
    "home_xg_for",
    "home_xg_against",
    "away_xg_for", 
    "away_xg_against",
    "league_position_diff",
    "form_difference",
    "home_advantage"
]

def main():
    """Main function to inspect model features"""
    print("\n🔍 ML Model Feature Inspector")
    print("="*70)
    
    # Specify your model path
    model_path = "models/xg_predictor_eastern_202506.pkl"
    
    # Create inspector
    inspector = MLFeatureInspector(model_path)
    
    # Load model
    inspector.load_model()
    
    # Set feature names (adjust based on your actual features)
    # If your model training saved feature names, you might load them here
    inspector.set_feature_names(EXAMPLE_FEATURE_NAMES[:inspector.model.n_features_in_])
    
    # Inspect model structure
    inspector.inspect_model_structure()
    
    # Create documentation
    inspector.create_feature_documentation()
    
    print("\n✅ Feature inspection complete!")

if __name__ == "__main__":
    main()