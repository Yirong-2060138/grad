# models/neural_network.py
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from helper import load_labeled_data, get_feature_names, compute_permutation_importance

def train_mlp(
    use_raw_features: bool = True,
    use_autoencoder_features: bool = True,
    include_extra_features: bool = True,
    calculate_importance: bool = True
) -> Dict[str, float]:
    """
    Trains and evaluates a Multilayer Perceptron (MLP) classifier on the labeled cloud detection data.
    
    Args:
        use_raw_features (bool): If True, includes basic engineered features
        use_autoencoder_features (bool): If True, includes autoencoder features 
        include_extra_features (bool): If True, includes extra gradient features
        calculate_importance (bool): Whether to calculate feature importance
    
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
    
    print(f"\nTraining MLP with {feature_str} features...")
    
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
    
    # Define MLP architecture
    # Adjust hidden layer sizes based on the input dimension
    hidden_layer_sizes = (min(100, X_train.shape[1] * 2), 
                         min(50, X_train.shape[1]), 
                         25)
    
    print(f"Using MLP with hidden layers: {hidden_layer_sizes}")
    
    # Initialize MLP model
    mlp_model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=0.0001,  # L2 regularization
        batch_size='auto',
        learning_rate="adaptive",
        max_iter=300,
        early_stopping=True, 
        validation_fraction=0.1,  
        n_iter_no_change=10, 
        random_state=42,
        verbose=True
    )
    
    # Train the model
    print("Training MLP model...")
    mlp_model.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = mlp_model.predict(X_test)
    y_proba = mlp_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    
    # Print classification report
    print("\nMLP Classification Report (Test Set):")
    print(classification_report(y_test, y_pred))
    print(f"MLP Test AUC: {test_auc:.3f}")
    
    # Compute feature importance if requested
    if calculate_importance:
        feature_names = get_feature_names(
            use_raw_features=use_raw_features,
            use_autoencoder_features=use_autoencoder_features,
            include_extra_features=include_extra_features,
            num_features=X_train.shape[1]
        )
        
        # Compute permutation importance
        df_importance = compute_permutation_importance(
            mlp_model, X_test, y_test, 
            feature_names, 
            n_repeats=10, 
            random_state=42
        )
    
        print("\nTop 20 features by permutation importance:")
        for i, row in df_importance.head(20).iterrows():
            print(f"{i+1}. {row['Feature']}: {row['Importance_Mean']:.4f} ± {row['Importance_StdDev']:.4f}")
    
    return {
        "model": f"MLP ({feature_str})",
        "test_auc": test_auc
    }

if __name__ == "__main__":

    result = train_mlp(
        use_raw_features=True,
        use_autoencoder_features=True,
        include_extra_features=True,
        calculate_importance=True
    )