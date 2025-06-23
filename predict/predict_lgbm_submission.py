import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.utils import FP_LIST, FP_DIMS, setup_logging, validate_config, load_test_data, prepare_submission_data, apply_platt_scaling, perform_diversification_clustering, filter_clusters_iteratively, assign_selection_labels, to_csr
from models.lgbm_ensemble import LGBMEnsemble

def parse_prediction_args():
    """Parse command line arguments for prediction script."""
    parser = argparse.ArgumentParser(description='DREAM Challenge WDR91 LightGBM Prediction Script')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data file (e.g., data/test_data.parquet)')
    parser.add_argument('--output-file', type=str, required=True, help='Path to output submission CSV file')
    parser.add_argument('--model-dir', type=str, default='./output/', help='Directory containing trained models (default: ./output/)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu). Default: auto-detect')
    return parser.parse_args()

class PredictionConfig:
    """Simple configuration class for prediction scripts."""
    def __init__(self, args):
        self.test_data = args.test_data
        self.output_file = args.output_file
        self.model_dir = args.model_dir
        self.seed = args.seed
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lgbm_device = "gpu" if self.device == "cuda" else "cpu"
        self.submission_template = args.test_data  # Use test data as template source
        self.outdir = args.model_dir  # Map model_dir to outdir for utils functions

def prepare_fingerprint_data(test_df):
    """
    Prepare fingerprint data for prediction by ensuring all required fingerprints are available.
    
    Args:
        test_df: Test dataframe containing fingerprint columns
        
    Returns:
        dict: Dictionary mapping fingerprint names to pandas Series
    """
    logging.info("Preparing fingerprint data for prediction...")
    
    fp_data = {}
    available_fps = []
    missing_fps = []
    
    for fp_name in FP_LIST:
        if fp_name in test_df.columns:
            fp_data[fp_name] = test_df[fp_name]
            available_fps.append(fp_name)
        else:
            # Create dummy fingerprints for missing ones (comma-separated zeros)
            fp_dim = FP_DIMS.get(fp_name, 2048)
            dummy_fp = ",".join(["0"] * fp_dim)
            fp_data[fp_name] = pd.Series([dummy_fp] * len(test_df))
            missing_fps.append(fp_name)
    
    logging.info("Available fingerprints: %s", available_fps)
    if missing_fps:
        logging.warning("Missing fingerprints (using dummy data): %s", missing_fps)
    
    return fp_data

def load_trained_model(cfg):
    """
    Load the trained LightGBM ensemble model from disk.
    
    Args:
        cfg: Configuration object containing output directory
        
    Returns:
        LGBMEnsemble: Loaded trained model
        
    Raises:
        FileNotFoundError: If model file cannot be found
        Exception: If model loading fails
    """
    model_path = os.path.join(cfg.model_dir, "lgbm_ensemble.pkl")
    
    logging.info("Loading trained model from: %s", model_path)
    
    if not os.path.exists(model_path):
        error_msg = f"LGBMEnsemble model not found: {model_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        lgbm_model = LGBMEnsemble.load(model_path)
        logging.info("Successfully loaded LGBMEnsemble model")
        logging.info("Model fingerprints: %s", lgbm_model.fingerprints)
        logging.info("Model device: %s", getattr(lgbm_model, 'device', 'N/A'))
          # Validate model has trained components
        if not lgbm_model.models or all(not models for models in lgbm_model.models.values()):
            raise ValueError("Loaded model appears to be empty or untrained")
        
        return lgbm_model
        
    except Exception as e:
        error_msg = f"Failed to load LGBMEnsemble model: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg) from e

def generate_predictions(lgbm_model, fp_data):
    """
    Generate predictions using the trained LightGBM ensemble model.
    
    Args:
        lgbm_model: Trained LGBMEnsemble model
        fp_data: Dictionary mapping fingerprint names to pandas Series
        
    Returns:
        np.ndarray: Prediction probabilities
    """
    logging.info("Generating predictions with LightGBM ensemble...")
    
    try:
        predictions = lgbm_model.predict(fp_data, to_csr)
        logging.info("Predictions generated successfully. Shape: %s", predictions.shape)
        logging.info("Prediction statistics - Min: %.4f, Max: %.4f, Mean: %.4f", 
                    predictions.min(), predictions.max(), predictions.mean())
        return predictions
        
    except Exception as e:        
        error_msg = f"Failed to generate predictions: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg) from e

def merge_ecfp4_fingerprints(top_compounds_df, original_test_df):
    """
    Merge ECFP4 fingerprints from original test data into top compounds dataframe.
    
    Args:
        top_compounds_df: DataFrame containing top compounds
        original_test_df: Original test dataframe with ECFP4 fingerprints
        
    Returns:
        pd.DataFrame: DataFrame with ECFP4 fingerprints merged
    """
    merged_df = top_compounds_df.merge(
        original_test_df[['RandomID', 'ECFP4']],
        on='RandomID',
        how='left'
    )
    
    logging.info("ECFP4 fingerprints merged into top compounds: %s", merged_df.shape)
    
    # Diagnostics for ECFP4 column
    logging.info("ECFP4 column exists: %s", 'ECFP4' in merged_df.columns)
    logging.info("Number of non-null ECFP4: %s", merged_df['ECFP4'].notnull().sum())
    
    if merged_df['ECFP4'].notnull().sum() > 0:
        example_values = merged_df['ECFP4'].dropna().head(5).tolist()
        value_lengths = merged_df['ECFP4'].dropna().apply(lambda x: len(x)).value_counts().head(10)
        logging.info("Example ECFP4 values: %s", example_values)
        logging.info("ECFP4 value lengths: %s", value_lengths.to_dict())
    
    return merged_df

def create_final_submission(submission_df, diversified_compounds, score_column='lgbm_p'):
    """
    Create final submission dataframe with selection labels for all compounds.
    
    Args:
        submission_df: Full submission dataframe
        diversified_compounds: DataFrame with diversified top compounds
        score_column: Column name for scores
        
    Returns:
        pd.DataFrame: Final submission dataframe with all required columns
    """
    # Get selected compound IDs
    sel_500_ids = set(diversified_compounds['RandomID'])
    sel_200_ids = set(diversified_compounds.head(200)['RandomID'])
    
    # Add selection labels to full submission
    submission_df['Sel_500'] = submission_df['RandomID'].apply(lambda x: 1 if x in sel_500_ids else 0)
    submission_df['Sel_200'] = submission_df['RandomID'].apply(lambda x: 1 if x in sel_200_ids else 0)
    
    # Create final submission format
    final_submission_df = submission_df[['RandomID', 'Sel_200', 'Sel_500', score_column]].rename(
        columns={score_column: 'Score'}
    )
    
    logging.info('Final submission DataFrame prepared: %s', final_submission_df.shape)
    logging.info('Selection counts - Sel_200: %s, Sel_500: %s', 
                final_submission_df['Sel_200'].sum(), final_submission_df['Sel_500'].sum())
    
    return final_submission_df

def main():
    args = parse_prediction_args()
    cfg = PredictionConfig(args)
    
    # Setup logging first
    log_file = setup_logging(cfg, "predict_lgbm")
    
    # Validate configuration
    if not validate_config(cfg, required_files=['test_data'], script_type="prediction"):
        logging.error("Configuration validation failed. Exiting.")
        return 1

    try:
        # Load test data
        test_df_final = load_test_data(cfg)
        
        # Create submission template from test data
        submission_df = test_df_final[['RandomID']].copy()
        logging.info("Created submission template from test data: %s", submission_df.shape)
        
        fp_data = prepare_fingerprint_data(test_df_final)
          # Load trained model
        lgbm = load_trained_model(cfg)
        
        # Generate predictions
        preds = generate_predictions(lgbm, fp_data)
        test_df_final["lgbm_p"] = preds
        
        # Apply calibration if available
        calibrated_preds, calibration_applied = apply_platt_scaling(cfg, preds)
        if calibration_applied:
            test_df_final["lgbm_p_calib"] = calibrated_preds
            # Use calibrated predictions for submission
            final_preds = calibrated_preds
            score_column = 'lgbm_p_calib'
            logging.info("Using calibrated predictions for submission")
        else:
            # Use original predictions if calibration failed
            final_preds = preds
            score_column = 'lgbm_p'
            logging.info("Using original predictions for submission (calibration not available)")
        
        # Prepare submission data
        submission_df = prepare_submission_data(submission_df, test_df_final, final_preds, score_column=score_column)
        
        # Extract top 1000 compounds for diversification
        top_1000_df = submission_df.head(1000).copy()
        logging.info("Top 1000 compounds extracted for diversification: %s", top_1000_df.shape)

        # Merge ECFP4 fingerprints for diversification
        original_test_df = pd.read_parquet(cfg.test_data)
        top_1000_df = merge_ecfp4_fingerprints(top_1000_df, original_test_df)

        # Perform diversification clustering to select diversified top 500
        clustered_df = perform_diversification_clustering(top_1000_df, ecfp4_column='ECFP4')
          # Filter clusters iteratively to maintain diversity
        diversified_top500 = filter_clusters_iteratively(clustered_df, max_per_cluster=2, target_size=500)
        
        # Assign selection labels
        diversified_top500 = assign_selection_labels(diversified_top500, score_column=score_column)
        
        # Create final submission dataframe
        final_submission_df = create_final_submission(submission_df, diversified_top500, score_column=score_column)
        
        # Ensure output directory exists before saving
        output_dir = os.path.dirname(cfg.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.info("Created output directory: %s", output_dir)
        
        # Save final submission file to custom output path
        final_submission_df.to_csv(cfg.output_file, index=False)
        logging.info("Final submission CSV saved to: %s", cfg.output_file)
        logging.info("Submission file shape: %s", final_submission_df.shape)
        logging.info("Submission columns: %s", final_submission_df.columns.tolist())
        
        logging.info("LGBM Prediction Pipeline Finished.")
        return 0
        
    except Exception as e:
        logging.error("Exception during LGBM prediction pipeline: %s", str(e), exc_info=True)
        try:
            error_path = os.path.join(cfg.model_dir, "test_set_prediction_error_lgbm.log")
            with open(error_path, "w", encoding="utf-8") as f:
                import traceback
                f.write(traceback.format_exc())
            logging.info("Error traceback written to: %s", error_path)
        except Exception as e2:
            logging.error("Failed to write error traceback: %s", str(e2))
        
        # Return error code for proper exit handling
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
