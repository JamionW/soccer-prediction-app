from autogluon.tabular import TabularPredictor
import pandas as pd
import json
from pathlib import Path

def inspect_autogluon_model(model_path: str):
    """Inspect an AutoGluon model comprehensively"""
    
    print(f"\n🔍 AutoGluon Model Inspector")
    print("="*70)
    print(f"Model path: {model_path}")
    
    # Load the predictor
    predictor = TabularPredictor.load(model_path)
    
    # Basic info
    print("\n📊 BASIC MODEL INFORMATION:")
    info = predictor.info()
    for key, value in info.items():
        if key != 'model_info':  # Skip the detailed model info for now
            print(f"   {key}: {value}")
    
    # Model names and types
    print("\n🤖 MODELS IN ENSEMBLE:")
    model_names = predictor.model_names()
    for i, name in enumerate(model_names, 1):
        print(f"   {i}. {name}")
    
    print(f"\n   Best model: {predictor.model_best}")
    
    # Feature metadata
    print("\n📋 FEATURE METADATA:")
    try:
        feature_metadata = predictor.feature_metadata
        print(f"   Number of features: {len(feature_metadata)}")
        print(f"\n   Features by type:")
        for dtype, features in feature_metadata.type_map_raw.items():
            if features:
                print(f"      {dtype}: {len(features)} features")
                for feat in list(features)[:3]:  # Show first 3
                    print(f"         - {feat}")
                if len(features) > 3:
                    print(f"         ... and {len(features) - 3} more")
    except Exception as e:
        print(f"   Could not get feature metadata: {e}")
    
    # Training configuration
    print("\n⚙️ TRAINING CONFIGURATION:")
    try:
        # Load the metadata file directly
        metadata_path = Path(model_path) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"   AutoGluon version: {metadata.get('version', 'Unknown')}")
            print(f"   Training time: {metadata.get('time_fit', 'Unknown')} seconds")
            print(f"   Preset: {metadata.get('hyperparameters', {}).get('default', {}).get('AG_args_fit', {}).get('preset', 'Unknown')}")
    except Exception as e:
        print(f"   Could not load metadata: {e}")
    
    # Model performance summary
    print("\n📈 MODEL PERFORMANCE SUMMARY:")
    try:
        # Get the leaderboard
        leaderboard = predictor.leaderboard(silent=True)
        print("\n   Model Leaderboard:")
        print(leaderboard[['model', 'score_val', 'pred_time_val', 'fit_time', 'stack_level']].to_string())
    except Exception as e:
        print(f"   Could not get leaderboard: {e}")
    
    # Feature importance (without test data)
    print("\n🎯 FEATURE IMPORTANCE (from training):")
    try:
        # Try to get feature importance without test data
        importance = predictor.feature_importance()
        print(f"\n   Top 15 most important features:")
        for i, (feature, score) in enumerate(importance.head(15).items(), 1):
            print(f"   {i:2d}. {feature:<30} {score:.4f}")
    except Exception as e:
        print(f"   Feature importance requires test data. Skipping.")
    
    # Save detailed report
    report_path = Path("output") / "autogluon_model_report.txt"
    with open(report_path, 'w') as f:
        f.write("AutoGluon Model Detailed Report\n")
        f.write("="*70 + "\n")
        f.write(f"Model path: {model_path}\n\n")
        
        f.write("Model Information:\n")
        f.write(json.dumps(info, indent=2))
        f.write("\n\nModel Names:\n")
        for name in model_names:
            f.write(f"- {name}\n")
    
    print(f"\n📄 Detailed report saved to: {report_path}")

def extract_feature_list(model_path: str):
    """Extract and save the list of features used"""
    
    predictor = TabularPredictor.load(model_path)
    
    # Get feature metadata
    feature_metadata = predictor.feature_metadata
    all_features = list(feature_metadata.get_features())
    
    # Create a feature documentation file
    doc_content = f"""# AutoGluon Model Features

Model: {model_path}
Total Features: {len(all_features)}

## Feature List:

"""
    
    # Group features by type
    type_map = feature_metadata.type_map_raw
    
    for dtype, features in type_map.items():
        if features:
            doc_content += f"\n### {dtype} Features ({len(features)}):\n"
            for feat in sorted(features):
                doc_content += f"- {feat}\n"
    
    # Save documentation
    doc_path = Path("output") / "autogluon_features.md"
    with open(doc_path, 'w') as f:
        f.write(doc_content)
    
    print(f"\n📋 Feature documentation saved to: {doc_path}")
    
    # Also save as JSON for programmatic use
    feature_dict = {
        "total_features": len(all_features),
        "features": all_features,
        "features_by_type": {str(k): list(v) for k, v in type_map.items() if v}
    }
    
    json_path = Path("output") / "autogluon_features.json"
    with open(json_path, 'w') as f:
        json.dump(feature_dict, f, indent=2)
    
    print(f"📋 Feature JSON saved to: {json_path}")

if __name__ == "__main__":
    model_path = "models/xg_predictor_eastern_202506.pkl"
    
    # Inspect the model
    inspect_autogluon_model(model_path)
    
    # Extract feature list
    extract_feature_list(model_path)
    
    print("\n✅ Model inspection complete!")