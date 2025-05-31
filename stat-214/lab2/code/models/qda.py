# models/qda.py
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from helper import load_labeled_data, get_feature_names, compute_permutation_importance

def train_qda(
    use_raw_features: bool = True,
    use_autoencoder_features: bool = True,
    include_extra_features: bool = True,
    calculate_importance: bool = True,
    reg_param: float = 0.0  # Regularization parameter
) -> Dict[str, float]:
    """
    Trains and evaluates a Quadratic Discriminant Analysis classifier on the labeled cloud detection data.
    
    Args:
        use_raw_features (bool): If True, includes basic engineered features
        use_autoencoder_features (bool): If True, includes autoencoder features 
        include_extra_features (bool): If True, includes extra gradient features
        calculate_importance (bool): Whether to calculate feature importance
        reg_param (float): Regularization parameter for QDA (0.0 means no regularization)
    
    Returns:
        Dict[str, float]: A dictionary containing model details and performance metrics
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
    
    print(f"\nTraining QDA with {feature_str} features...")
    
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
    
    # Initialize QDA model
    # Use regularization to handle potential singularity issues in covariance matrices
    qda_model = QuadraticDiscriminantAnalysis(
        reg_param=reg_param,
        store_covariance=True  # Store covariance matrices for inspection if needed
    )
    
    # Train the model
    print("Training QDA model...")
  
    qda_model.fit(X_train, y_train)
    training_successful = True
    
    # Generate predictions
    y_pred = qda_model.predict(X_test)
    y_proba = qda_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    
    # Print classification report
    print("\nQDA Classification Report (Test Set):")
    print(classification_report(y_test, y_pred))
    print(f"QDA Test AUC: {test_auc:.3f}")
    
    # Compute feature importance if requested
    if calculate_importance:
        # Get feature names
        feature_names = get_feature_names(
            use_raw_features=use_raw_features,
            use_autoencoder_features=use_autoencoder_features,
            include_extra_features=include_extra_features,
            num_features=X_train.shape[1]
        )
        
        # Compute permutation importance
        df_importance = compute_permutation_importance(
            qda_model, X_test, y_test, 
            feature_names, 
            n_repeats=10, 
            random_state=42
        )
        
        # Print top features
        print("\nTop 20 features by permutation importance:")
        for i, row in df_importance.head(20).iterrows():
            print(f"{i+1}. {row['Feature']}: {row['Importance_Mean']:.4f} ± {row['Importance_StdDev']:.4f}")
    
    return {
        "model": f"QDA ({feature_str})",
        "test_auc": test_auc,
        "status": "success"
    }

if __name__ == "__main__":
    result = train_qda(
        use_raw_features=True,
        use_autoencoder_features=True,
        include_extra_features=True,
        calculate_importance=True,
        reg_param=0.01  # Use some regularization to avoid singularity issues
    )