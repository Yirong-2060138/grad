# models/logistic_regression.py
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
# Import the helper functions we created
from helper import load_labeled_data, get_feature_names, compute_permutation_importance

def train_logistic_regression(
    use_raw_features: bool = True,
    use_autoencoder_features: bool = False,
    include_extra_features: bool = False,
    calculate_importance: bool = False
) -> Dict[str, float]:
    """
    Trains and evaluates a Logistic Regression model on the labeled cloud detection data.
    
    Depending on the feature flags, this function loads raw features, autoencoder features,
    or a combination of both. The data is split into training, validation, and test sets,
    and additional gradient features are computed if requested.
    
    Args:
        use_raw_features (bool, optional): 
            If True, uses engineered raw features. Defaults to True.
        use_autoencoder_features (bool, optional):
            If True, uses precomputed autoencoder latent features. Defaults to False.
        include_extra_features (bool, optional):
            If True, computes and includes additional gradient features. Defaults to False.
        calculate_importance (bool, optional):
            If True, calculates feature importance using permutation. Only for all features.
    
    Returns:
        Dict[str, float]: A dictionary containing:
            - 'model': Name of the model.
            - 'val_auc': ROC-AUC score on the validation set.
            - 'test_auc': ROC-AUC score on the test set.
            - 'feature_importance': Feature importance data (if calculate_importance=True)
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
    
    print(f"\nTraining logistic regression with {feature_str} features...")
    
    # Load labeled data with the appropriate feature sets
    X_train, X_val, X_test, y_train, y_val, y_test = load_labeled_data(
        use_raw_features=use_raw_features,
        use_autoencoder_features=use_autoencoder_features,
        include_extra_features=include_extra_features
    )
    
    # Print feature dimensions for debugging
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train the model
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"Validation AUC: {val_auc:.3f}")
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    
    print("\nLogistic Regression Classification Report (Test Set):")
    print(classification_report(y_test, y_pred))
    print(f"Test AUC: {test_auc:.3f}")
    
    # Return results dictionary
    result = {
        "model": f"Logistic Regression ({feature_str})",
        "val_auc": val_auc,
        "test_auc": test_auc
    }
    
    # Calculate feature importance if requested
    if calculate_importance:
        # Get feature names using the helper function
        feature_names = get_feature_names(
            use_raw_features=use_raw_features,
            use_autoencoder_features=use_autoencoder_features,
            include_extra_features=include_extra_features,
            num_features=X_train.shape[1]
        )
        
        # Compute permutation importance using the helper function
        df_importance = compute_permutation_importance(
            model, X_test, y_test, 
            feature_names, 
            n_repeats=10, 
            random_state=42,
            model_name="Logistic Regression"
        )
        
        # Store feature importance in the result
        importance_dict = {
            "feature_names": df_importance['Feature'].tolist(),
            "importance_mean": df_importance['Importance_Mean'].tolist(),
            "importance_std": df_importance['Importance_StdDev'].tolist()
        }
        result["feature_importance"] = importance_dict
        
        # If we want to look at the coefficients for logistic regression
        if hasattr(model, 'coef_'):
            # Get absolute coefficients
            coef = np.abs(model.coef_[0])
            # Sort indices by coefficient magnitude
            sorted_idx = coef.argsort()[::-1]
            
            plt.figure(figsize=(14, 10))
            
            # Limit to top 30 features for clarity
            top_n = min(30, len(sorted_idx))
            top_idx = sorted_idx[:top_n]
            
            # Get feature names and values for plot
            feat_names = [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in top_idx]
            coef_values = coef[top_idx]
            
            # Sort for better visualization
            indices = np.argsort(coef_values)
            sorted_names = [feat_names[i] for i in indices]
            sorted_coefs = coef_values[indices]
            
            # Plot using matplotlib's barh
            plt.barh(range(top_n), sorted_coefs, color="lightgreen")
            plt.yticks(range(top_n), sorted_names)
            plt.xlabel("Coefficient Magnitude")
            plt.title("Top Feature Importance for Logistic Regression (Coefficient Magnitude)")
            plt.tight_layout()
            plt.savefig(f"logreg_coefficient_{feature_str.replace(' + ', '_')}.png", dpi=300)
            print(f"Coefficient plot saved as logreg_coefficient_{feature_str.replace(' + ', '_')}.png")
            
            # Print top coefficients
            print("\nTop 20 features by coefficient magnitude:")
            for i in range(min(20, len(sorted_idx))):
                idx = sorted_idx[i]
                feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                coef_val = coef[idx]
                print(f"{i+1}. {feat_name}: {coef_val:.4f}")
    
    return result

def evaluate_feature_combinations():
    """
    Evaluates all combinations of feature types and compares their performance.
    """
    results = []
    
    # Test different feature combinations
    combinations = [
        {"use_raw_features": True, "use_autoencoder_features": False, "include_extra_features": False},
        {"use_raw_features": True, "use_autoencoder_features": False, "include_extra_features": True},
        {"use_raw_features": False, "use_autoencoder_features": True, "include_extra_features": False},
        {"use_raw_features": True, "use_autoencoder_features": True, "include_extra_features": False},
        {"use_raw_features": True, "use_autoencoder_features": True, "include_extra_features": True}
    ]
    
    for config in combinations:
        result = train_logistic_regression(**config)
        results.append(result)
    
    # Print summary of results
    print("\n===== SUMMARY OF RESULTS =====")
    for result in results:
        print(f"{result['model']}: Test AUC = {result['test_auc']:.3f}, Val AUC = {result['val_auc']:.3f}")
    
    return results

def evaluate_with_feature_importance():
    """
    Runs the model with all features enabled and calculates feature importance.
    """
    print("\n===== EVALUATING MODEL WITH ALL FEATURES + IMPORTANCE ANALYSIS =====")
    result = train_logistic_regression(
        use_raw_features=True,
        use_autoencoder_features=True,
        include_extra_features=True,
        calculate_importance=True
    )
    
    return result

if __name__ == "__main__":
    # evaluate_feature_combinations()

    evaluate_with_feature_importance()