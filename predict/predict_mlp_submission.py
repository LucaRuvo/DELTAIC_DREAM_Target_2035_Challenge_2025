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

from utils.utils import FP_LIST, FP_DIMS, ensure, fp_string_to_tensor, setup_logging, validate_config, load_test_data, prepare_submission_data, perform_diversification_clustering, filter_clusters_iteratively, assign_selection_labels
from models.multibranch_mlp import MultiBranchMLP
from sklearn.linear_model import LogisticRegression

def parse_prediction_args():
    """Parse command line arguments for MLP prediction script."""
    parser = argparse.ArgumentParser(description='DREAM Challenge WDR91 MLP Prediction Script')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data file (e.g., data/test_data.parquet)')
    parser.add_argument('--output-file', type=str, required=True, help='Path to output submission CSV file')
    parser.add_argument('--model-dir', type=str, default='./output/', help='Directory containing trained models (default: ./output/)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for prediction (default: 256)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu). Default: auto-detect')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

class PredictionConfig:
    """Simple configuration class for prediction scripts.""" 
    def __init__(self, args):
        self.test_data = args.test_data
        self.output_file = args.output_file
        self.model_dir = args.model_dir
        self.batch_size = args.batch_size
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = args.seed        # Add default MLP configuration for model loading (must match training config)
        self.branch_hidden_dims = [2048, 1024]
        self.branch_embedding_dim = 256
        self.common_hidden_dims = [1024, 512]
        self.dropout_rate = 0.4
        self.n_splits = 5  # Default for model loading
        
        # For backwards compatibility with utils functions
        self.submission_template = args.test_data  # Use test data as template source
        self.outdir = args.model_dir  # Map model_dir to outdir for utils functions

def prepare_fingerprint_data_mlp(test_df, device):
    """
    Prepare fingerprint data for MLP prediction by converting to PyTorch tensors.
    
    Args:
        test_df: Test dataframe containing fingerprint columns
        device: PyTorch device (CPU or CUDA)
        
    Returns:
        dict: Dictionary mapping fingerprint names to PyTorch tensors
    """
    logging.info("Preparing fingerprint data for MLP prediction...")
    
    fp_data = {}
    available_fps = []
    missing_fps = []
    
    for fp_name in FP_LIST:
        if fp_name in test_df.columns:
            fp_data[fp_name] = fp_string_to_tensor(test_df[fp_name], fp_name, device=device)
            available_fps.append(fp_name)
        else:
            # Create dummy tensors for missing fingerprints
            fp_dim = FP_DIMS.get(fp_name, 2048)
            fp_data[fp_name] = torch.zeros((len(test_df), fp_dim), dtype=torch.uint8, device=device)
            missing_fps.append(fp_name)
    
    logging.info("Available fingerprints: %s", available_fps)
    if missing_fps:
        logging.warning("Missing fingerprints (using dummy tensors): %s", missing_fps)
    
    return fp_data

def load_mlp_fold_models(cfg):
    """
    Load all fold models for MLP ensemble prediction.
    
    Args:
        cfg: Configuration object containing model parameters and paths
        
    Returns:
        list: List of loaded PyTorch models
        
    Raises:
        ValueError: If no valid models are found
    """
    logging.info("Loading MLP fold models...")
    
    models = []
    missing_models = []
    
    for fold in range(1, cfg.n_splits + 1):
        model_path = os.path.join(cfg.model_dir, f"best_model_fold{fold}.pt")
        
        if not os.path.exists(model_path):
            logging.warning(f"Model for fold {fold} not found: {model_path}")
            missing_models.append(fold)
            continue
        
        try:
            # Initialize model architecture
            model = MultiBranchMLP(
                fp_input_dims=FP_DIMS,
                branch_hidden_dims=cfg.branch_hidden_dims,
                branch_embedding_dim=cfg.branch_embedding_dim,
                common_hidden_dims=cfg.common_hidden_dims,
                dropout_rate=cfg.dropout_rate,
                fp_list=FP_LIST
            ).to(cfg.device)
            
            # Load trained weights
            model.load_state_dict(torch.load(model_path, map_location=cfg.device))
            model.eval()
            
            models.append(model)
            logging.info(f"Successfully loaded model for fold {fold}")
            
        except Exception as e:
            logging.error(f"Failed to load model for fold {fold}: {str(e)}")
            missing_models.append(fold)
    
    if not models:
        raise ValueError(f"No valid fold models found. Missing folds: {missing_models}")
    
    logging.info(f"Loaded {len(models)} out of {cfg.n_splits} fold models")
    if missing_models:
        logging.warning(f"Missing models for folds: {missing_models}")
    
    return models

def generate_mlp_predictions(models, fp_data, device, batch_size=1024):
    """
    Generate predictions using MLP fold models with batched inference and averaging.
    
    Args:
        models: List of loaded PyTorch models
        fp_data: Dictionary mapping fingerprint names to PyTorch tensors
        device: PyTorch device (CPU or CUDA)
        batch_size: Batch size for inference (default: 1024)
        
    Returns:
        np.ndarray: Averaged prediction probabilities
    """
    logging.info(f"Generating MLP predictions using {len(models)} fold models...")
    
    # Get number of samples
    num_samples = len(next(iter(fp_data.values())))
    fold_preds = []
    
    for fold_idx, model in enumerate(models, 1):
        logging.info(f"Processing fold {fold_idx}/{len(models)}")
        
        preds = []
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                # Prepare batch data
                batch_fps = {
                    fp_name: fp_data[fp_name][i:i+batch_size].to(device) 
                    for fp_name in FP_LIST
                }
                
                # Forward pass
                outputs = model(batch_fps)
                
                # Apply sigmoid and convert to numpy
                batch_preds = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds.append(batch_preds)
        
        # Concatenate all batch predictions for this fold
        fold_predictions = np.concatenate(preds)
        fold_preds.append(fold_predictions)
        
        logging.info(f"Fold {fold_idx} predictions completed. Shape: {fold_predictions.shape}")
    
    # Average predictions across all folds
    final_predictions = np.mean(np.vstack(fold_preds), axis=0)
    
    logging.info("MLP predictions generated successfully")
    logging.info("Prediction statistics - Min: %.4f, Max: %.4f, Mean: %.4f", 
                final_predictions.min(), final_predictions.max(), final_predictions.mean())
    
    return final_predictions

def apply_platt_scaling_mlp(cfg, predictions):
    """
    Apply Platt scaling (probability calibration) to MLP predictions using Excel-based OOF data.
    
    Args:
        cfg: Configuration object containing output directory
        predictions: Raw prediction probabilities from the MLP model
        
    Returns:
        tuple: (calibrated_predictions, calibration_applied)
            - calibrated_predictions: Calibrated probabilities (or original if calibration failed)
            - calibration_applied: Boolean indicating if calibration was successfully applied
    """
    logging.info("Attempting to apply Platt scaling calibration for MLP...")
    
    # MLP uses Excel format for OOF predictions
    oof_path_excel = os.path.join(cfg.outdir, "training_summary.xlsx")
    
    if not os.path.exists(oof_path_excel):
        logging.warning("OOF predictions Excel file not found: %s. Skipping calibration.", oof_path_excel)
        return predictions, False
    
    try:
        # Load OOF predictions from Excel
        oof_df = pd.read_excel(oof_path_excel, sheet_name="OOF_Predictions")
        logging.info("OOF predictions loaded from Excel for calibration. Shape: %s", oof_df.shape)
        
        # Validate required columns
        required_columns = ['pred_proba', 'true_label']
        missing_columns = [col for col in required_columns if col not in oof_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in OOF data: {missing_columns}")
        
        # Prepare calibration data
        oof_preds = oof_df["pred_proba"].values
        oof_labels = oof_df["true_label"].values
        oof_mask = ~np.isnan(oof_preds)
        
        if oof_mask.sum() < 10:
            raise ValueError(f"Insufficient valid OOF predictions for calibration: {oof_mask.sum()}")
          # Fit Platt scaler
        platt_scaler = LogisticRegression(solver='lbfgs', max_iter=1000)
        platt_scaler.fit(oof_preds[oof_mask].reshape(-1, 1), oof_labels[oof_mask])
        
        # Apply calibration
        calibrated_preds = platt_scaler.predict_proba(predictions.reshape(-1, 1))[:, 1]
        
        logging.info("Platt scaling applied successfully for MLP")
        logging.info("Calibrated prediction statistics - Min: %.4f, Max: %.4f, Mean: %.4f", 
                    calibrated_preds.min(), calibrated_preds.max(), calibrated_preds.mean())
        
        return calibrated_preds, True
        
    except Exception as e:
        logging.warning("Could not apply Platt scaling for MLP: %s. Using original predictions.", str(e))
        return predictions, False

def merge_ecfp4_fingerprints_mlp(top_compounds_df, original_test_df):
    """
    Merge ECFP4 fingerprints from original test data into top compounds dataframe for MLP.
    
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
    
    logging.info("ECFP4 fingerprints merged into top MLP compounds: %s", merged_df.shape)
    
    # Diagnostics for ECFP4 column
    logging.info("ECFP4 column exists: %s", 'ECFP4' in merged_df.columns)
    logging.info("Number of non-null ECFP4: %s", merged_df['ECFP4'].notnull().sum())
    
    if merged_df['ECFP4'].notnull().sum() > 0:
        example_values = merged_df['ECFP4'].dropna().head(5).tolist()
        value_lengths = merged_df['ECFP4'].dropna().apply(lambda x: len(x)).value_counts().head(10)
        logging.info("Example ECFP4 values: %s", example_values)
        logging.info("ECFP4 value lengths: %s", value_lengths.to_dict())
    
    return merged_df

def create_final_mlp_submission(submission_df, diversified_compounds, score_column='mlp_p'):
    """
    Create final submission dataframe with selection labels for all compounds (MLP version).
    
    Args:
        submission_df: Full submission dataframe
        diversified_compounds: DataFrame with diversified top compounds
        score_column: Column name for scores (default: 'mlp_p')
        
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
    
    logging.info('Final MLP submission DataFrame prepared: %s', final_submission_df.shape)
    logging.info('Selection counts - Sel_200: %s, Sel_500: %s', 
                final_submission_df['Sel_200'].sum(), final_submission_df['Sel_500'].sum())
    
    return final_submission_df

def main():
    args = parse_prediction_args()
    cfg = PredictionConfig(args)
    
    # Setup logging first
    log_file = setup_logging(cfg, "predict_mlp")
    
    # Ensure output directory exists
    ensure(cfg.model_dir)
    
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
        
        fp_data = prepare_fingerprint_data_mlp(test_df_final, cfg.device)
          # Load trained models
        models = load_mlp_fold_models(cfg)
        
        # Generate predictions
        final_test_preds_proba = generate_mlp_predictions(models, fp_data, cfg.device, cfg.batch_size)
        test_df_final["mlp_p"] = final_test_preds_proba
        logging.info("Predictions added to test dataframe: %s", test_df_final.shape)
        
        calibrated_preds, calibration_applied = apply_platt_scaling_mlp(cfg, final_test_preds_proba)
        if calibration_applied:
            test_df_final["mlp_p_calib"] = calibrated_preds
            # Use calibrated predictions for submission
            final_preds = calibrated_preds
            score_column = 'mlp_p_calib'
            logging.info("Using calibrated predictions for submission")
        else:
            # Use original predictions if calibration failed
            final_preds = final_test_preds_proba
            score_column = 'mlp_p'
            logging.info("Using original predictions for submission (calibration not available)")
            
        # Prepare submission data
        submission_df = prepare_submission_data(submission_df, test_df_final, final_preds, score_column=score_column)
        
        # Extract top 1000 compounds for diversification
        top_1000_df = submission_df.head(1000).copy()
        logging.info("Top 1000 compounds extracted for diversification: %s", top_1000_df.shape)

        # Merge ECFP4 fingerprints for diversification
        original_test_df = pd.read_parquet(cfg.test_data)
        top_1000_df = merge_ecfp4_fingerprints_mlp(top_1000_df, original_test_df)        # Perform diversification clustering to select diversified top 500
        clustered_df = perform_diversification_clustering(top_1000_df, ecfp4_column='ECFP4')
        
        # Filter clusters iteratively to maintain diversity
        diversified_top500 = filter_clusters_iteratively(clustered_df, max_per_cluster=2, target_size=500)
        
        # Assign selection labels
        diversified_top500 = assign_selection_labels(diversified_top500, score_column=score_column)
        
        # Create final submission dataframe
        final_submission_df = create_final_mlp_submission(submission_df, diversified_top500, score_column=score_column)
        
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
        
        logging.info("MLP Prediction Pipeline Finished.")
        return 0
        
    except Exception as e:
        logging.error("Exception during MLP prediction pipeline: %s", str(e), exc_info=True)
        try:
            error_path = os.path.join(cfg.model_dir, "test_set_prediction_error_mlp.log")
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
