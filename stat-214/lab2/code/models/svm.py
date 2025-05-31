# models/svm.py
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from helper import load_labeled_data, compute_permutation_importance, get_feature_names

def train_svm(
    use_raw_features: bool = True,
    use_autoencoder_features: bool = True,
    include_extra_features: bool = True,
    calculate_importance: bool = True,
    sample_size: int = None 
) -> Dict[str, float]:
    """
    Trains and evaluates an SVM classifier on the labeled cloud detection data.
    
    Args:
        use_raw_features (bool): If True, includes basic engineered features
        use_autoencoder_features (bool): If True, includes autoencoder features 
        include_extra_features (bool): If True, includes extra gradient features
        calculate_importance (bool): Whether to calculate feature importance
        sample_size (int, optional): If provided, limits the training data to this number of samples
    
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
    
    print(f"\nTraining SVM with {feature_str} features...")
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_labeled_data(
        use_raw_features=use_raw_features,
        use_autoencoder_features=use_autoencoder_features,
        include_extra_features=include_extra_features,
        test_size=0.2, 
        val_size=0.2, 
        random_state=42
    )
    
    print(f"Original training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    X_train_sample = X_train
    y_train_sample = y_train
    
    print(f"Training data shape after sampling: {X_train_sample.shape}")
    
    # Initialize SVM model
    # Use linear kernel for high-dimensional data (better for feature importance)
    kernel = "linear" if X_train_sample.shape[1] > 20 else "rbf"
    print(f"Using SVM with {kernel} kernel")
    
    svm_model = SVC(
        kernel=kernel,
        probability=True,
        C=1.0,
        class_weight='balanced',
        random_state=42
    )
    
    # Train the model
    print("Training SVM model...")
    svm_model.fit(X_train_sample, y_train_sample)
    
    # Generate predictions
    y_pred = svm_model.predict(X_test)
    y_proba = svm_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    
    # Print classification report
    print("\nSVM Classification Report (Test Set):")
    print(classification_report(y_test, y_pred))
    print(f"SVM Test AUC: {test_auc:.3f}")
    
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
            svm_model, X_test, y_test, 
            feature_names, 
            n_repeats=10, 
            random_state=42
        )
        
        print("\nTop 20 features by permutation importance:")
        for i, row in df_importance.head(20).iterrows():
            print(f"{i+1}. {row['Feature']}: {row['Importance_Mean']:.4f} ± {row['Importance_StdDev']:.4f}")
        
        # For linear SVM kernels, also print built-in feature weights
        if kernel == "linear" and hasattr(svm_model, 'coef_'):
            # Get absolute coefficients
            coef = np.abs(svm_model.coef_[0])
            # Create DataFrame
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coef
            }).sort_values('Coefficient', ascending=False)
            
            # Print top features by coefficient
            print("\nTop 20 features by SVM coefficient magnitude (linear kernel only):")
            for i, row in coef_df.head(20).iterrows():
                print(f"{i+1}. {row['Feature']}: {row['Coefficient']:.4f}")
            
            # Plot SVM coefficients
            plt.figure(figsize=(14, 10))
            top_n = min(30, len(coef_df))
            plot_df = coef_df.head(top_n).sort_values('Coefficient')
            
            plt.barh(range(top_n), plot_df['Coefficient'], color="lightsalmon")
            plt.yticks(range(top_n), plot_df['Feature'])
            plt.xlabel("Coefficient Magnitude")
            plt.title("Top Feature Importance for SVM Model (Coefficient Magnitude)")
            plt.tight_layout()
            plt.savefig("svm_coefficient_importance.png", dpi=300)
            print("SVM coefficient plot saved as svm_coefficient_importance.png")
    
    return {
        "model": f"SVM ({feature_str})",
        "test_auc": test_auc
    }

if __name__ == "__main__":
    result = train_svm(
        use_raw_features=True,
        use_autoencoder_features=True,
        include_extra_features=True,
        calculate_importance=True,
        sample_size=10000  # Use 10,000 training samples - adjust as needed
    )