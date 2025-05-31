# models/xgboost_model.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from helper import load_labeled_data, get_feature_names, compute_permutation_importance


def train_xgboost(
    use_raw_features: bool = True,
    use_autoencoder_features: bool = True,
    include_extra_features: bool = True
) -> Dict[str, float]:
    """
    Trains and evaluates an XGBoost classifier on the labeled cloud detection data.
    Uses all available features and computes permutation feature importance.
    
    Args:
        use_raw_features (bool): If True, includes basic engineered features
        use_autoencoder_features (bool): If True, includes autoencoder features 
        include_extra_features (bool): If True, includes extra gradient features
    
    Returns:
        dict: A dictionary containing the model name and ROC-AUC score.
    """
    # Create a feature description for reporting
    feature_desc = []
    if use_raw_features:
        feature_desc.append("raw")
    if use_autoencoder_features:
        feature_desc.append("autoencoder")
    if include_extra_features and use_raw_features:
        feature_desc.append("extra engineered")
    feature_str = " + ".join(feature_desc)
    
    print(f"\nTraining XGBoost with {feature_str} features...")
    
    # Load labeled data with all features
    X_train, X_val, X_test, y_train, y_val, y_test = load_labeled_data(
        use_raw_features=use_raw_features,
        use_autoencoder_features=use_autoencoder_features,
        include_extra_features=include_extra_features,
        test_size=0.2, 
        val_size=0.2, 
        random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Set up XGBoost hyperparameters
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'tree_method': 'hist',  # Faster training method
        'random_state': 42
    }
    

    xgb_model = XGBClassifier(**params)

    eval_set = [(X_val, y_val)]
    xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
    
    # Generate predictions
    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    
    # Print classification report
    print("\nXGBoost Classification Report (Test Set):")
    print(classification_report(y_test, y_pred))
    print(f"XGBoost Test AUC: {test_auc:.3f}")
    
    # Get feature names
    feature_names = get_feature_names(
        use_raw_features=use_raw_features,
        use_autoencoder_features=use_autoencoder_features,
        include_extra_features=include_extra_features,
        num_features=X_train.shape[1]
    )
    
    # Compute and plot permutation feature importance
    print("\nCalculating permutation feature importance...")
    df_importance = compute_permutation_importance(
        xgb_model, X_test, y_test, 
        feature_names, 
        n_repeats=10, 
        random_state=42
    )
    
    # Print top features
    print("\nTop 20 features by permutation importance:")
    for i, row in df_importance.head(20).iterrows():
        print(f"{i+1}. {row['Feature']}: {row['Importance_Mean']:.4f} ± {row['Importance_StdDev']:.4f}")
    
    # Plot feature importance from the model itself (for comparison)
    if hasattr(xgb_model, 'feature_importances_'):
        plt.figure(figsize=(14, 10))
        
        # Get feature importances directly from model
        model_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Plot top features
        top_n = min(30, len(model_importance))
        plot_df = model_importance.head(top_n).sort_values('Importance')
        
        plt.barh(range(top_n), plot_df['Importance'], color="lightblue")
        plt.yticks(range(top_n), plot_df['Feature'])
        plt.xlabel("Feature Importance")
        plt.title("XGBoost Built-in Feature Importance")
        plt.tight_layout()
        plt.savefig("xgboost_builtin_importance.png", dpi=300)
        print("Built-in feature importance plot saved as xgboost_builtin_importance.png")
    
    return {
        "model": f"XGBoost ({feature_str})",
        "test_auc": test_auc
    }

if __name__ == "__main__":
    # Train XGBoost with all features and compute permutation importance
    result = train_xgboost(
        use_raw_features=True,
        use_autoencoder_features=True,
        include_extra_features=True
    )