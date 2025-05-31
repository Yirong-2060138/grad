import os
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


def compute_ndai_gradients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes gradients of the NDAI feature with respect to X and Y coordinates,
    along with the gradient magnitude and its logarithm.
    """
    df = df.copy()  # Create a copy to avoid modifying the original dataframe
    df["NDAI_X_gradient"] = df.groupby("Y")["NDAI"].diff() / df.groupby("Y")["X"].diff()
    df["NDAI_Y_gradient"] = df.groupby("X")["NDAI"].diff() / df.groupby("X")["Y"].diff()
    df["NDAI_X_gradient"] = df["NDAI_X_gradient"].fillna(0).replace([np.inf, -np.inf], 0)
    df["NDAI_Y_gradient"] = df["NDAI_Y_gradient"].fillna(0).replace([np.inf, -np.inf], 0)
    df["NDAI_gradient_magnitude"] = np.sqrt(df["NDAI_X_gradient"]**2 + df["NDAI_Y_gradient"]**2)
    epsilon = 1e-6
    df["NDAI_log_gradient_magnitude"] = np.log(df["NDAI_gradient_magnitude"] + epsilon)
    return df

def compute_feature_ndais_and_gradients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all NDAI indices and their gradients for all feature channels (AF, BF, CF).
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing radiance values
    
    Returns:
        pd.DataFrame: DataFrame with all NDAI features and their gradients
    """
    df = df.copy() 
    
    # Compute NDAI for each feature channel
    for channel in ["AF", "BF", "CF"]:
        df[f"NDAI_{channel}"] = (df["Radiance_DF"] - df[f"Radiance_{channel}"]) / (df["Radiance_DF"] + df[f"Radiance_{channel}"])
    
    # Compute gradients for each NDAI
    for channel in ["AF", "BF", "CF"]:
        prefix = f"{channel}NDAI"
        ndai_col = f"NDAI_{channel}"
        
        df[f"{prefix}_X_gradient"] = df.groupby("Y")[ndai_col].diff() / df.groupby("Y")["X"].diff()
        df[f"{prefix}_Y_gradient"] = df.groupby("X")[ndai_col].diff() / df.groupby("X")["Y"].diff()
        
        df[f"{prefix}_X_gradient"] = df[f"{prefix}_X_gradient"].fillna(0).replace([np.inf, -np.inf], 0)
        df[f"{prefix}_Y_gradient"] = df[f"{prefix}_Y_gradient"].fillna(0).replace([np.inf, -np.inf], 0)
        
        df[f"{prefix}_gradient_magnitude"] = np.sqrt(df[f"{prefix}_X_gradient"]**2 + df[f"{prefix}_Y_gradient"]**2)
        epsilon = 1e-6
        df[f"{prefix}_log_gradient_magnitude"] = np.log(df[f"{prefix}_gradient_magnitude"] + epsilon)
    
    return df

def load_labeled_data(
    use_raw_features: bool = True,
    use_autoencoder_features: bool = False,
    include_extra_features: bool = False,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads labeled cloud detection data and integrates different feature types based on specified parameters.
    
    Parameters:
        use_raw_features (bool): Whether to include basic raw features (NDAI, SD, CORR, Radiance_* etc.)
        use_autoencoder_features (bool): Whether to load and merge autoencoder features.
        include_extra_features (bool): Whether to compute and include additional gradient features.
        test_size (float): Fraction of the total data reserved for the test set.
        val_size (float): Fraction of the remaining data to be used as the validation set.
        random_state (int): Seed for reproducibility.
    
    Returns:
        Tuple containing:
            - X_train (np.ndarray): Training set features.
            - X_val (np.ndarray): Validation set features.
            - X_test (np.ndarray): Test set features.
            - y_train (np.ndarray): Training set labels.
            - y_val (np.ndarray): Validation set labels.
            - y_test (np.ndarray): Test set labels.
    """
    
    # Path configurations
    data_dir = "../../data/image_data"
    outcome_dir = "../Outcome_3_image_2"
    
    labeled_files = ["O013257.npz", "O013490.npz", "O012791.npz"]
    ae_mapping = {
        "O013257.npz": "image15_ae.csv",
        "O013490.npz": "image17_ae.csv",
        "O012791.npz": "image18_ae.csv"
    }

    raw_base_cols = ["NDAI", "SD", "CORR", "Radiance_DF", "Radiance_CF",
                     "Radiance_BF", "Radiance_AF", "Radiance_AN"]
    
    gradient_cols = ["NDAI_X_gradient", "NDAI_Y_gradient",
                     "NDAI_gradient_magnitude", "NDAI_log_gradient_magnitude"]
    
    extra_cols = []
    for channel in ["AF", "BF", "CF"]:
        prefix = f"{channel}NDAI"
        extra_cols.extend([
            f"NDAI_{channel}",
            f"{prefix}_X_gradient", 
            f"{prefix}_Y_gradient",
            f"{prefix}_gradient_magnitude", 
            f"{prefix}_log_gradient_magnitude"
        ])
    
    # Initialize data containers
    X_list = []
    y_list = []
    
    # Process each file
    for file in labeled_files:
        print(f"Processing {file}...")
        npz_path = os.path.join(data_dir, file)

        # Load NPZ data

        data = np.load(npz_path)["arr_0"]
        columns = ["Y", "X", "NDAI", "SD", "CORR", "Radiance_DF", 
                "Radiance_CF", "Radiance_BF", "Radiance_AF", "Radiance_AN", "expert_label"]
        df = pd.DataFrame(data, columns=columns)

        # Filter for labeled pixels and round coordinates
        df = df[df["expert_label"].isin([1, -1])].reset_index(drop=True)
        df["Y"] = df["Y"].round(0)
        df["X"] = df["X"].round(0)
        
        original_count = df.shape[0]
        print(f"  Found {original_count} labeled pixels in {file}")
        
        # Compute features based on selected options
        if use_raw_features:
            df = compute_ndai_gradients(df)
            if include_extra_features:
                df = compute_feature_ndais_and_gradients(df)
        
        # Handle autoencoder features if requested
        if use_autoencoder_features:
            csv_filename = ae_mapping.get(file)
            
            csv_path = os.path.join(outcome_dir, csv_filename)
                
            try:
                # Load autoencoder features
                df_ae = pd.read_csv(csv_path)
                df_ae.rename(columns={"y": "Y", "x": "X"}, inplace=True)
                df_ae["Y"] = df_ae["Y"].round(0)
                df_ae["X"] = df_ae["X"].round(0)
                
                # Merge with main dataframe
                df_merged = pd.merge(df, df_ae, on=["Y", "X"], how="inner")
                
                df = df_merged
            except Exception as e:
                print(f"Error loading/merging autoencoder features from {csv_path}: {e}")
        
        # Extract labels (convert expert_label 1 to 1, -1 to 0)
        labels = (df["expert_label"] == 1).astype(int).values
        
        # Build feature matrix based on selected options
        feature_cols = []
        
        if use_raw_features:
            feature_cols.extend(raw_base_cols)
            feature_cols.extend(gradient_cols)
            
            if include_extra_features:
                feature_cols.extend(extra_cols)
        
        if use_autoencoder_features:
            ae_cols = [col for col in df.columns if col.startswith("ae")]
            feature_cols.extend(ae_cols)
        
        # Make sure all requested columns exist in the dataframe
        existing_cols = [col for col in feature_cols if col in df.columns]
        
        # Extract features
        X_img = df[existing_cols].values

        
        X_list.append(X_img)
        y_list.append(labels)
        
        print(f"  Added {X_img.shape[0]} samples with {X_img.shape[1]} features")
    
    
    # Stack all data
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    print(f"Total dataset: {X_all.shape[0]} samples with {X_all.shape[1]} features")

    # Split into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Split training+validation into training and validation sets
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val_scaled, y_train_val, test_size=val_fraction, 
        stratify=y_train_val, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    return X_train, X_val, X_test_scaled, y_train, y_val, y_test

def get_feature_names(
    use_raw_features: bool,
    use_autoencoder_features: bool,
    include_extra_features: bool,
    num_features: int
) -> List[str]:
    """
    Generate feature names based on the selected feature configuration.
    
    Args:
        use_raw_features: If True, includes raw feature names
        use_autoencoder_features: If True, includes autoencoder feature names
        include_extra_features: If True, includes extra engineered feature names
        num_features: Total number of features to validate
    
    Returns:
        list: List of feature names
    """
    feature_names = []
    
    if use_raw_features:
        # Base raw features
        feature_names.extend([
            "NDAI", "SD", "CORR", "Radiance_DF", "Radiance_CF",
            "Radiance_BF", "Radiance_AF", "Radiance_AN"
        ])
    
        feature_names.extend([
            "NDAI_X_gradient", "NDAI_Y_gradient",
            "NDAI_gradient_magnitude", "NDAI_log_gradient_magnitude"
        ])
        
        if include_extra_features:

            for channel in ["AF", "BF", "CF"]:
                feature_names.append(f"NDAI_{channel}")
            
                prefix = f"{channel}NDAI"
                feature_names.extend([
                    f"{prefix}_X_gradient",
                    f"{prefix}_Y_gradient",
                    f"{prefix}_gradient_magnitude",
                    f"{prefix}_log_gradient_magnitude"
                ])
    
    if use_autoencoder_features:
        # Estimate the number of autoencoder features
        ae_count = num_features - len(feature_names)
        feature_names.extend([f"ae_{i}" for i in range(ae_count)])
    
    return feature_names

def compute_permutation_importance(
    model, X_test: np.ndarray, y_test: np.ndarray, 
    feature_names: List[str], 
    n_repeats: int = 10, 
    random_state: int = 42,
    model_name: str = "model"
) -> pd.DataFrame:
    """
    Computes permutation feature importance for a trained model.
    
    Args:
        model: Trained model with predict_proba method
        X_test: Test data features
        y_test: Test data labels
        feature_names: List of feature names
        n_repeats: Number of times to permute a feature
        random_state: Random seed for reproducibility
        model_name: Name of the model for plot titles and filenames
    
    Returns:
        pd.DataFrame: DataFrame with feature importance results
    """
    print(f"Computing permutation importance for {model_name}... This may take a while...")
    
    # Calculate permutation importance
    result = permutation_importance(
        model, X_test, y_test, 
        n_repeats=n_repeats, 
        random_state=random_state,
        scoring='roc_auc',
        n_jobs=-1 
    )
    
    # Convert to DataFrame for easier analysis
    df_importance = pd.DataFrame({
        'Feature': feature_names if len(feature_names) == X_test.shape[1] else [f"Feature_{i}" for i in range(X_test.shape[1])],
        'Importance_Mean': result.importances_mean,
        'Importance_StdDev': result.importances_std
    })
    
    # Sort by importance (descending)
    df_importance = df_importance.sort_values('Importance_Mean', ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(14, 10))
    
    # Limit to top 30 features for clarity
    top_n = min(30, len(df_importance))
    plot_df = df_importance.head(top_n).sort_values('Importance_Mean')
    
    # Plot using matplotlib's barh (horizontal bar chart)
    plt.barh(range(top_n), plot_df['Importance_Mean'], 
             xerr=plot_df['Importance_StdDev'], 
             color="skyblue", ecolor="black", capsize=5)
    plt.yticks(range(top_n), plot_df['Feature'])
    plt.xlabel("Mean AUC decrease")
    plt.title(f"Top Feature Importance for {model_name} (Permutation)")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_feature_importance.png", dpi=300)
    print(f"Feature importance plot saved as {model_name.lower().replace(' ', '_')}_feature_importance.png")
    
    # Print top features
    print(f"\nTop 20 features by permutation importance for {model_name}:")
    for i, row in df_importance.head(20).iterrows():
        print(f"{i+1}. {row['Feature']}: {row['Importance_Mean']:.4f} ± {row['Importance_StdDev']:.4f}")
    
    return df_importance