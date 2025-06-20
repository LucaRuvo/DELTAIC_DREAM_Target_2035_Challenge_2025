import os
import sys
import logging
import random
import warnings
import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering

# Conditional imports
try:
    from rdkit import DataStructs
    from rdkit.DataStructs.cDataStructs import ExplicitBitVect
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

FP_LIST = ["ATOMPAIR", "MACCS", "ECFP6", "ECFP4", "FCFP4", "FCFP6", "TOPTOR", "RDK", "AVALON"]
FP_DIMS = {
    "MACCS": 167,
    "ATOMPAIR": 2048,
    "ECFP6": 2048,
    "ECFP4": 2048,
    "FCFP4": 2048,
    "FCFP6": 2048,
    "TOPTOR": 2048,
    "RDK": 2048,
    "AVALON": 2048
}
FP_DIM = 2048

def get_branch_hidden_dims(preset):
    """
    Get hidden layer dimensions for neural network branches based on preset.
    
    Args:
        preset: String preset name ('deeper', 'wider', 'shallower', or default)
        
    Returns:
        list: Hidden layer dimensions [dim1, dim2]
    """
    if preset.lower() == 'deeper':
        return [2048, 1024]
    elif preset.lower() == 'wider':
        return [2048, 2048]
    elif preset.lower() == 'shallower':
        return [512, 256]
    else:
        return [1024, 512]

def ensure(p):
    """
    Create directory if it doesn't exist (like mkdir -p).
    
    Args:
        p: Path to directory to create
    """
    if not os.path.exists(p):
        os.makedirs(p)

def set_seed(s):
    """
    Set random seeds for reproducibility across random, numpy, and torch.
    
    Args:
        s: Random seed value
    """
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)

def tensorize_fp(arr, device=None):
    """
    Convert numpy array to PyTorch tensor for fingerprint data.
    
    Args:
        arr: Numpy array to convert
        device: PyTorch device (cuda/cpu), defaults to None for auto-detection
        
    Returns:
        torch.Tensor: Converted tensor with uint8 dtype
    """
    t = torch.tensor(arr, dtype=torch.uint8, device=device, requires_grad=False)
    return t

def fp_string_to_tensor(fp_series, fp_name="ECFP4", device=None):
    num_samples = len(fp_series)
    fp_dim = FP_DIMS.get(fp_name, FP_DIM)
    fp_array = np.zeros((num_samples, fp_dim), dtype=np.uint8)
    for i, fp_str in enumerate(fp_series.fillna("").values):
        if fp_str is None or (isinstance(fp_str, str) and fp_str.strip() == ""):
            continue
        try:
            if isinstance(fp_str, str):
                values = [int(float(x)) for x in fp_str.split(",")]
            elif isinstance(fp_str, (np.ndarray, list)):
                values = np.array(fp_str, dtype=int).flatten()
            else:
                values = np.zeros(fp_dim, dtype=int)
            current_fp_len = len(values)
            if current_fp_len > fp_dim:
                fp_array[i, :fp_dim] = values[:fp_dim]
            else:
                fp_array[i, :current_fp_len] = values
        except Exception as e:
            logging.warning("fp_string_to_tensor error: %s, type=%s", str(e), type(fp_str).__name__)
            pass
    return tensorize_fp(fp_array, device=device)

def parse_ecfp4_binary(fp_str, n_bits=2048):
    if not isinstance(fp_str, str):
        return np.zeros(n_bits, dtype=np.uint8)
    bits = [int(b) if b in {'0', '1'} else 0 for b in fp_str.strip().split(',')]
    if len(bits) != n_bits:
        bits = (bits + [0]*n_bits)[:n_bits]
    return np.array(bits, dtype=np.uint8)

def binarize_fp_string(fp_str):
    try:
        if isinstance(fp_str, str):
            arr = np.array(fp_str.split(','), dtype=int)
        elif isinstance(fp_str, (np.ndarray, list)):
            arr = np.array(fp_str, dtype=int).flatten()
        else:
            arr = np.zeros(2048, dtype=int)
        arr_bin = (arr > 0).astype(int)
        return ','.join(map(str, arr_bin))
    except Exception as e:
        logging.warning(f"binarize_fp_string error: {e}, type={type(fp_str)}")
        return ','.join(['0']*2048)

def to_csr(series: pd.Series) -> sp.csr_matrix:
    """Fingerprint counts 2048-dim CSR (row-major)"""
    rows, cols, data = [], [], []
    for i, txt in enumerate(series.fillna("").values):
        # Handle string fingerprints
        if isinstance(txt, str):
            if txt.strip() == "":
                continue
            fp_items = txt.split(",")
        # Handle numpy arrays or lists
        elif isinstance(txt, (np.ndarray, list)):
            if len(txt) == 0:
                continue
            fp_items = txt
        # Handle anything else as empty
        else:
            continue
        for j, v in enumerate(map(int, fp_items)):
            if v:
                rows.append(i); cols.append(j); data.append(v)
    return sp.csr_matrix((data, (rows, cols)),
                         shape=(len(series), 2048), dtype=np.float32)

def calculate_plate_ppv(y_true, y_pred_proba, k=138):
    """Calculates Precision@k (specifically for top 138 predictions)."""
    df = pd.DataFrame({'true': y_true, 'pred': y_pred_proba})
    # Handle cases where there are fewer than k samples
    actual_k = min(k, len(df))
    if actual_k == 0:
        return 0.0
    top_k = df.nlargest(actual_k, 'pred')
    if len(top_k) == 0: # Should not happen if actual_k > 0, but safety check
        return 0.0
    precision = top_k['true'].sum() / len(top_k)
    return precision

def setup_logging(cfg, script_name="prediction"):
    """
    Setup logging configuration for any pipeline script.
    
    Args:
        cfg: Configuration object containing output directory
        script_name: Name of the script for log file naming (default: "prediction")
        
    Returns:
        str: Path to the log file
    """
    ensure(cfg.outdir)
    log_file = os.path.join(cfg.outdir, f"{script_name}.log")
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s", 
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'), 
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("=" * 50)
    logging.info("%s Pipeline Started", script_name.title())
    logging.info("=" * 50)
    logging.info("Configuration: %s", str(vars(cfg)))
    logging.info("Using device: %s", getattr(cfg, 'device', 'N/A'))
    logging.info("Log file: %s", log_file)
    
    return log_file

def validate_config(cfg, required_files=None, script_type="prediction"):
    """
    Validate configuration parameters for pipeline scripts.
    
    Args:
        cfg: Configuration object containing all parameters
        required_files: List of required file attributes (e.g., ['test_data', 'submission_template'])
        script_type: Type of script for customized validation (default: "prediction")
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    validation_errors = []
    
    # Default required files for prediction scripts
    if required_files is None:
        if script_type == "prediction":
            required_files = ['test_data', 'submission_template']
        elif script_type == "training":
            required_files = ['train_data']
        else:
            required_files = []
    
    # Check required file paths
    for file_attr in required_files:
        file_path = getattr(cfg, file_attr, None)
        if not file_path:
            validation_errors.append(f"{file_attr.replace('_', ' ').title()} path is required. Use --{file_attr.replace('_', '-')} argument.")
        elif not os.path.exists(file_path):
            validation_errors.append(f"{file_attr.replace('_', ' ').title()} file not found: {file_path}")
    
    # Check output directory is valid
    if not getattr(cfg, 'outdir', None):
        validation_errors.append("Output directory is required. Use --outdir argument.")
    
    # Log validation results
    if validation_errors:
        logging.error("Configuration validation failed:")
        for error in validation_errors:
            logging.error("  - %s", error)
        return False
    
    logging.info("Configuration validation passed")
    for file_attr in required_files:
        file_path = getattr(cfg, file_attr, None)
        if file_path:
            logging.info("%s: %s", file_attr.replace('_', ' ').title(), file_path)
    logging.info("Output directory: %s", cfg.outdir)
    
    return True

def load_submission_template(cfg):
    """
    Load and validate the submission template CSV file.
    
    Args:
        cfg: Configuration object containing submission_template path
        
    Returns:
        pd.DataFrame: Loaded submission template
        
    Raises:
        FileNotFoundError: If submission template file cannot be loaded
    """
    try:
        submission_df = pd.read_csv(cfg.submission_template)
        logging.info("Submission template loaded: %s", submission_df.shape)
        logging.info("Submission template columns: %s", submission_df.columns.tolist())
        
        # Validate required columns
        required_columns = ['RandomID']
        missing_columns = [col for col in required_columns if col not in submission_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in submission template: {missing_columns}")
        
        return submission_df
    except Exception as e:
        logging.error("Failed to load submission template: %s", str(e))
        raise

def prepare_submission_data(submission_df, test_df, predictions, score_column='prediction_score'):
    """
    Prepare submission data by merging predictions with submission template and ranking compounds.
    
    Args:
        submission_df: Submission template DataFrame
        test_df: Test data DataFrame containing RandomID
        predictions: Prediction probabilities array
        score_column: Name of the column for storing prediction scores (default: 'prediction_score')
        
    Returns:
        pd.DataFrame: Ranked submission DataFrame with predictions merged
    """
    logging.info("Preparing submission data...")
    
    # Add predictions to test dataframe
    test_df_with_preds = test_df.copy()
    test_df_with_preds[score_column] = predictions
    
    # Merge predictions with submission template
    merged_df = submission_df.merge(
        test_df_with_preds[['RandomID', score_column]],
        on='RandomID',
        how='left'
    )
    logging.info("Predictions merged into submission template: %s", merged_df.shape)
    
    # Check for missing predictions
    missing_preds = merged_df[score_column].isna().sum()
    if missing_preds > 0:
        logging.warning("Found %d compounds with missing predictions", missing_preds)
        # Fill missing predictions with 0 (lowest probability)
        merged_df[score_column] = merged_df[score_column].fillna(0.0)
    
    # Rank compounds by prediction score
    ranked_df = merged_df.sort_values(score_column, ascending=False).reset_index(drop=True)
    logging.info("Compounds ranked by prediction score: %s", ranked_df.shape)
    logging.info("Top prediction scores: %s", ranked_df[score_column].head(10).tolist())
    
    return ranked_df

def apply_platt_scaling(cfg, predictions):
    """
    Apply Platt scaling (probability calibration) to raw predictions if OOF data is available.
    
    Args:
        cfg: Configuration object containing output directory
        predictions: Raw prediction probabilities from the model
        
    Returns:
        tuple: (calibrated_predictions, calibration_applied)
            - calibrated_predictions: Calibrated probabilities (or original if calibration failed)
            - calibration_applied: Boolean indicating if calibration was successfully applied
    """
    logging.info("Attempting to apply Platt scaling calibration...")
    
    # Try Excel file first, then fallback to CSV
    oof_path_excel = os.path.join(cfg.outdir, "training_summary_lgbm.xlsx")
    oof_path_csv = os.path.join(cfg.outdir, "OOF_Predictions.csv")
    
    oof_df = None
    
    # Try to load from Excel file (preferred)
    if os.path.exists(oof_path_excel):
        try:
            oof_df = pd.read_excel(oof_path_excel, sheet_name='OOF_Predictions')
            logging.info("OOF predictions loaded from Excel file. Shape: %s", oof_df.shape)
        except Exception as e:
            logging.warning("Failed to load OOF predictions from Excel: %s", str(e))
    
    # Fallback to CSV file
    if oof_df is None and os.path.exists(oof_path_csv):
        try:
            oof_df = pd.read_csv(oof_path_csv)
            logging.info("OOF predictions loaded from CSV file. Shape: %s", oof_df.shape)
        except Exception as e:
            logging.warning("Failed to load OOF predictions from CSV: %s", str(e))
    
    # If no OOF data found, skip calibration
    if oof_df is None:
        logging.warning("OOF predictions not found in %s or %s. Skipping calibration.", oof_path_excel, oof_path_csv)
        return predictions, False

    try:
        
        # Validate required columns
        required_columns = ['pred_proba', 'true_label']
        missing_columns = [col for col in required_columns if col not in oof_df.columns]
        if missing_columns:
            logging.warning("Missing columns in OOF file: %s. Skipping calibration.", missing_columns)
            return predictions, False
        
        # Prepare calibration data
        oof_preds = oof_df["pred_proba"].values
        oof_labels = oof_df["true_label"].values
        oof_mask = ~np.isnan(oof_preds)
        
        if oof_mask.sum() < 10:  # Need minimum samples for calibration
            logging.warning("Insufficient valid OOF predictions (%d) for calibration. Skipping.", oof_mask.sum())
            return predictions, False
        
        # Fit Platt scaler
        platt_scaler = LogisticRegression(solver='lbfgs', max_iter=1000)
        platt_scaler.fit(oof_preds[oof_mask].reshape(-1, 1), oof_labels[oof_mask])
        
        # Apply calibration
        calibrated_preds = platt_scaler.predict_proba(predictions.reshape(-1, 1))[:, 1]
        
        logging.info("Platt scaling applied successfully")
        logging.info("Calibrated prediction statistics - Min: %.4f, Max: %.4f, Mean: %.4f", 
                    calibrated_preds.min(), calibrated_preds.max(), calibrated_preds.mean())
        
        return calibrated_preds, True
        
    except Exception as e:
        logging.warning("Could not apply Platt scaling: %s. Using original predictions.", str(e))
        return predictions, False

def convert_ecfp4_to_bitvector(ecfp4_string, expected_len=2048):
    """
    Convert ECFP4 string representation to RDKit ExplicitBitVect.
    
    Args:
        ecfp4_string: String representation of ECFP4 fingerprint (comma-separated)
        expected_len: Expected length of the fingerprint (default 2048)
        
    Returns:
        ExplicitBitVect or None: RDKit bitvector or None if conversion fails    """
    if not RDKIT_AVAILABLE:
        logging.warning("RDKit not available, cannot convert ECFP4 to bitvector")
        return None
        
    try:
        if pd.notnull(ecfp4_string) and isinstance(ecfp4_string, str):
            arr = [int(i) for i in ecfp4_string.split(",") if i.strip() != ""]
            if len(arr) == expected_len:
                bv = ExplicitBitVect(expected_len)
                for idx, v in enumerate(arr):
                    if v != 0:
                        bv.SetBit(idx)
                return bv
    except Exception:
        return None
    return None

def is_valid_fingerprint(fp, expected_len=2048):
    """
    Check if a fingerprint bitvector is valid.
    
    Args:
        fp: RDKit bitvector
        expected_len: Expected length of the fingerprint
        
    Returns:
        bool: True if valid, False otherwise
    """
    return fp is not None and fp.GetNumBits() == expected_len

def calculate_tanimoto_similarity_matrix(fingerprints):
    """
    Calculate Tanimoto similarity matrix for a list of fingerprints.
    
    Args:
        fingerprints: List of RDKit bitvectors
        
    Returns:
        np.ndarray: Similarity matrix (n x n)
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit not available, cannot calculate Tanimoto similarity")
        
    n = len(fingerprints)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    
    return sim_matrix

def perform_diversification_clustering(df, ecfp4_column='ECFP4', distance_threshold=0.32):
    """
    Perform diversification clustering based on ECFP4 fingerprints.
    
    Args:
        df: DataFrame containing ECFP4 fingerprints
        ecfp4_column: Name of the column containing ECFP4 data
        distance_threshold: Distance threshold for clustering
        
    Returns:
        pd.DataFrame: DataFrame with added 'Cluster' column
        
    Raises:        ValueError: If insufficient valid fingerprints are found
    """
    
    warnings.filterwarnings('ignore')
    
    # Convert ECFP4 strings to bitvectors
    df_copy = df.copy()
    df_copy['ECFP4_fp'] = df_copy[ecfp4_column].apply(convert_ecfp4_to_bitvector)
    
    # Extract valid fingerprints
    valid_fps = [(i, fp) for i, fp in enumerate(df_copy['ECFP4_fp']) if is_valid_fingerprint(fp)]
    
    if len(valid_fps) < 500:
        raise ValueError(f"Not enough valid ECFP4 fingerprints of length 2048 for clustering "
                        f"(found {len(valid_fps)}, need 500). Check your input data.")
    
    indices = [i for i, _ in valid_fps]
    fps = [fp for _, fp in valid_fps]
    
    logging.info(f"Processing {len(fps)} valid fingerprints for clustering")
    
    # Calculate similarity matrix
    sim_matrix = calculate_tanimoto_similarity_matrix(fps)
    
    # Convert to distance matrix
    dist_matrix = 1 - sim_matrix
      # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        metric='precomputed', 
        linkage='complete', 
        distance_threshold=distance_threshold
    )
    cluster_labels = clustering.fit_predict(dist_matrix)
    
    # Map cluster labels back to original indices
    cluster_mapping = {idx: label for idx, label in zip(indices, cluster_labels)}
    df_copy['Cluster'] = df_copy.index.map(lambda i: cluster_mapping.get(i, -1))
    
    logging.info(f"Clustering completed. Found {len(set(cluster_labels))} clusters")
    
    return df_copy

def filter_clusters_iteratively(df, max_per_cluster=2, target_size=500, max_iterations=10):
    """
    Iteratively filter clusters to maintain max compounds per cluster while reaching target size.
    
    Args:
        df: DataFrame with 'Cluster' column
        max_per_cluster: Maximum number of compounds per cluster
        target_size: Target number of compounds to maintain
        max_iterations: Maximum number of filtering iterations
        
    Returns:
        pd.DataFrame: Filtered DataFrame with diversified selection    """
    
    # Start with first target_size compounds
    current_df = df.head(target_size).copy()
    removed = set()
    
    for iteration in range(max_iterations):
        counts = current_df['Cluster'].value_counts()
        large_clusters = [c for c, cnt in counts.items() if cnt > max_per_cluster]
        
        if not large_clusters:
            logging.info(f"Clustering filter converged after {iteration} iterations")
            break
        
        # Identify compounds to remove from large clusters
        to_remove = []
        for cluster_id in large_clusters:
            cluster_rows = current_df[current_df['Cluster'] == cluster_id]
            # Keep first max_per_cluster, remove the rest
            to_remove.extend(cluster_rows.index[max_per_cluster:].tolist())
        
        removed.update(to_remove)
        
        # Keep compounds not marked for removal
        keep = [i for i in current_df.index if i not in removed]
        need = target_size - len(keep)
        
        # Refill from remaining compounds in original dataframe
        refill = [i for i in df.index if i not in keep and i not in removed][:need]
        current_df = df.loc[keep + refill].copy().reset_index(drop=True)
        
        logging.info(f"Iteration {iteration + 1}: Removed {len(to_remove)} compounds, "
                    f"refilled {len(refill)} compounds")
    
    # Add diversified rank
    diversified_df = current_df.copy()
    diversified_df['DiversifiedRank'] = np.arange(1, len(diversified_df) + 1)
    
    logging.info(f"Diversified selection completed: {len(diversified_df)} compounds")
    
    return diversified_df

def assign_selection_labels(df, score_column='prediction_score', top_200=200, top_500=500):
    """
    Assign Sel_200 and Sel_500 selection labels based on scores.
    
    Args:
        df: DataFrame with score column
        score_column: Name of column containing scores for ranking (default: 'prediction_score')
        top_200: Number of compounds for Sel_200
        top_500: Number of compounds for Sel_500
        
    Returns:
        pd.DataFrame: DataFrame with Sel_200 and Sel_500 columns added
    """    # Sort by score (descending) and reset index
    df_sorted = df.sort_values(score_column, ascending=False).reset_index(drop=True)
    
    # Initialize selection columns (default: not selected)
    df_sorted['Sel_500'] = 0
    df_sorted['Sel_200'] = 0
    
    # Assign top 500 labels
    df_sorted.loc[:top_500-1, 'Sel_500'] = 1
    
    # Assign top 200 labels  
    df_sorted.loc[:top_200-1, 'Sel_200'] = 1
      # Verify label counts
    sel_200_count = df_sorted['Sel_200'].sum()
    sel_500_count = df_sorted['Sel_500'].sum()
    
    logging.info(f"Selection labels assigned - Sel_200: {sel_200_count}, Sel_500: {sel_500_count}")
    
    # Validate expected counts (adjust for available data)
    available_compounds = len(df_sorted)
    expected_200 = min(top_200, available_compounds)
    expected_500 = min(top_500, available_compounds)
    
    if sel_200_count != expected_200:
        raise ValueError(f"Sel_200 count is {sel_200_count}, expected {expected_200}")
    if sel_500_count != expected_500:
        raise ValueError(f"Sel_500 count is {sel_500_count}, expected {expected_500}")
    
    return df_sorted

def save_submission_file(submission_df, cfg, filename="submission_lgbm.csv"):
    """
    Save the final submission dataframe to CSV file.
    
    Args:
        submission_df: Final submission dataframe
        cfg: Configuration object containing output directory
        filename: Name of the output file (default: "submission_lgbm.csv")
        
    Returns:
        str: Path to the saved file
        
    Raises:        Exception: If file saving fails
    """
    
    try:
        submission_csv_path = os.path.join(cfg.outdir, filename)
        submission_df.to_csv(submission_csv_path, index=False)
        
        logging.info(f'Final submission CSV saved to: {submission_csv_path}')
        logging.info(f'Submission file shape: {submission_df.shape}')
        logging.info(f'Submission columns: {submission_df.columns.tolist()}')
        
        # Validate the saved file
        if os.path.exists(submission_csv_path):
            file_size = os.path.getsize(submission_csv_path)
            logging.info(f'File size: {file_size} bytes')
        else:
            raise FileNotFoundError(f"Failed to create submission file: {submission_csv_path}")
        
        return submission_csv_path
        
    except Exception as e:
        error_msg = f"Failed to save submission file: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg) from e

def load_test_data(cfg):
    """
    Load and validate the test data file.
    
    Args:
        cfg: Configuration object containing test_data path
        
    Returns:
        pd.DataFrame: Loaded test dataset
        
    Raises:
        FileNotFoundError: If test data file cannot be loaded
        ValueError: If required columns are missing
    """
    try:
        test_df = pd.read_parquet(cfg.test_data)
        logging.info("Test set loaded: %s", test_df.shape)
        logging.info("Test set columns: %s", test_df.columns.tolist())
        
        # Validate required columns
        required_columns = ['RandomID']
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in test data: {missing_columns}")
        
        return test_df
    except Exception as e:
        logging.error("Failed to load test data: %s", str(e))
        raise