import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.cluster import MiniBatchKMeans

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import Config, parse_args
from utils.utils import (
    FP_LIST, ensure, set_seed, binarize_fp_string, parse_ecfp4_binary, 
    to_csr, setup_logging, validate_config
)
from models.lgbm_ensemble import LGBMEnsemble

def load_and_preprocess_training_data(cfg):
    """
    Load and preprocess training data with ECFP4 binarization.
    
    Args:
        cfg: Configuration object containing train_data path
        
    Returns:
        pd.DataFrame: Preprocessed training dataframe
        
    Raises:
        RuntimeError: If ECFP4 column processing fails
    """
    try:
        logging.info("Loading training data...")
        hitgen_df = pd.read_parquet(cfg.train_data)
        logging.info("Training data loaded: %s", hitgen_df.shape)
        
        # Binarize ECFP4 column for clustering
        if "ECFP4" in hitgen_df.columns:
            hitgen_df["ECFP4_binary"] = hitgen_df["ECFP4"].apply(binarize_fp_string)
            logging.info("ECFP4 column binarized successfully")
        elif "ECFP4_binary" not in hitgen_df.columns:
            raise RuntimeError("No ECFP4 or ECFP4_binary column found in input data.")
        
        # Validate required columns
        required_columns = ['LABEL']
        missing_columns = [col for col in required_columns if col not in hitgen_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in training data: {missing_columns}")
        
        logging.info("Training data preprocessing completed: %d rows", len(hitgen_df))
        return hitgen_df
        
    except Exception as e:
        logging.error("Failed to load and preprocess training data: %s", str(e))
        raise

def create_cluster_groups(hitgen_df, cfg):
    """
    Create cluster groups for stratified group k-fold cross-validation.
    
    Args:
        hitgen_df: Training dataframe
        cfg: Configuration object with grouping strategy
        
    Returns:
        pd.DataFrame: Dataframe with cluster_id column added
    """
    logging.info("Creating cluster groups for cross-validation...")
    
    grouping_strategy = cfg.grouping_strategy
    bb_cols = ['BB1_ID', 'BB2_ID', 'BB3_ID']
    has_bb_cols = all(col in hitgen_df.columns for col in bb_cols)
    use_bb = (grouping_strategy == 'bb') or (grouping_strategy == 'auto' and has_bb_cols)
    
    if use_bb:
        hitgen_df['cluster_id'] = (
            hitgen_df['BB1_ID'].astype(str) + '_' + 
            hitgen_df['BB2_ID'].astype(str) + '_' + 
            hitgen_df['BB3_ID'].astype(str)
        )
        logging.info("Using BB tuple grouping: %d unique groups", hitgen_df['cluster_id'].nunique())
    else:
        n_clusters = max(50, int(np.ceil(len(hitgen_df) / 50)))
        cluster_ids = pd.Series([0] * len(hitgen_df))
        
        try:
            fps = np.stack(hitgen_df["ECFP4_binary"].apply(parse_ecfp4_binary))
            km = MiniBatchKMeans(
                n_clusters=n_clusters, 
                batch_size=1024, 
                max_iter=100, 
                random_state=cfg.seed, 
                verbose=0
            )
            cluster_ids = km.fit_predict(fps)
            logging.info("MiniBatchKMeans clustering completed successfully")
        except Exception as e:
            logging.warning("MiniBatchKMeans failed: %s. Assigning random clusters.", str(e))
            cluster_ids = np.random.randint(0, n_clusters, size=len(hitgen_df))
            
        hitgen_df["cluster_id"] = cluster_ids
    
    logging.info("Using %s grouping strategy with %d unique clusters", 
                grouping_strategy, hitgen_df['cluster_id'].nunique())
    
    return hitgen_df

def prepare_balanced_dataset(hitgen_df, cfg):
    """
    Prepare balanced dataset by downsampling inactives to 1:10 ratio.
    
    Args:
        hitgen_df: Training dataframe with cluster_id
        cfg: Configuration object with random seed
        
    Returns:
        tuple: (train_df, folds) - balanced training dataframe and CV folds
    """
    logging.info("Preparing balanced dataset...")
    
    # Downsample inactives to 1:10 ratio
    actives = hitgen_df[hitgen_df["LABEL"] == 1]
    inactives = hitgen_df[hitgen_df["LABEL"] == 0]
    n_pos = len(actives)
    n_neg_target = 10 * n_pos
    
    if len(inactives) > n_neg_target:
        inactives_down = inactives.sample(n=n_neg_target, random_state=cfg.seed)
        logging.info("Downsampled inactives from %d to %d (target ratio 1:10)", 
                    len(inactives), len(inactives_down))
    else:
        inactives_down = inactives
        logging.info("Using all %d inactives (already below target ratio)", len(inactives))
    
    # Combine and shuffle
    train_df = pd.concat([actives, inactives_down], axis=0).sample(
        frac=1, random_state=cfg.seed
    ).reset_index(drop=True)
    
    # Create stratified group k-fold splits
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    folds = list(sgkf.split(train_df, y=train_df["LABEL"], groups=train_df["cluster_id"]))
    
    logging.info("Balanced training set prepared: %d rows, %s", 
                len(train_df), train_df['LABEL'].value_counts().to_dict())
    logging.info("Using %d folds for cross-validation", cfg.n_splits)
    
    return train_df, folds

def train_lgbm_ensemble(train_df, folds, cfg):
    """
    Train LGBM ensemble model with cross-validation.
    
    Args:
        train_df: Balanced training dataframe
        folds: Cross-validation fold indices
        cfg: Configuration object
          Returns:
        tuple: (LGBMEnsemble, labels) - trained ensemble model and true labels
    """
    logging.info("Training LGBM ensemble...")
    
    # Prepare fingerprint arrays
    X_dict = {fp: train_df[fp] for fp in FP_LIST if fp in train_df.columns}
    y = train_df["LABEL"].values    
    if not X_dict:
        raise ValueError("No valid fingerprint columns found in training data")
    
    logging.info("Using fingerprints: %s", list(X_dict.keys()))
    
    # Convert device format for LightGBM
    lgbm_device = "gpu" if cfg.device == "cuda" else "cpu"
    
    # Train ensemble
    lgbm = LGBMEnsemble(fingerprints=list(X_dict.keys()), device=lgbm_device, seed=cfg.seed)
    lgbm.train(X_dict, y, folds, to_csr)
    lgbm.fit_stacker(y)
    
    logging.info("LGBM ensemble training completed successfully")
    return lgbm, y

def save_training_results(lgbm, y_true, cfg):
    """
    Save trained model and training summary.
    
    Args:
        lgbm: Trained LGBM ensemble model
        y_true: True labels array for OOF predictions
        cfg: Configuration object
        
    Returns:
        str: Path to saved model file
    """
    logging.info("Saving training results...")
    
    # Save model
    model_path = os.path.join(cfg.outdir, "lgbm_ensemble.pkl")
    lgbm.save(model_path)
    logging.info("LGBM ensemble saved to: %s", model_path)
    
    # Collect metrics and OOF predictions for summary
    fold_metrics = []
    oof_records = []
    
    for fp in lgbm.fingerprints:
        # Per-fold metrics
        for m in lgbm.fold_metrics[fp]:
            fold_metrics.append(m)
          # OOF predictions
        oof = lgbm.oof_preds[fp]
        for i, (true_label, pred_proba) in enumerate(zip(y_true, oof)):
            oof_records.append({
                'index': i,
                'true_label': true_label,
                'pred_proba': pred_proba,
                'fingerprint': fp
            })
    
    # Save summary Excel and OOF predictions
    summary_path = os.path.join(cfg.outdir, 'training_summary_lgbm.xlsx')
    
    try:
        with pd.ExcelWriter(summary_path) as writer:
            # Configuration
            pd.DataFrame([vars(cfg)]).T.rename(columns={0: 'value'}).to_excel(
                writer, sheet_name='Config'
            )
            
            # Fold metrics
            pd.DataFrame(fold_metrics).to_excel(
                writer, sheet_name='Fold_Metrics', index=False
            )
            
            # OOF predictions
            oof_df = pd.DataFrame(oof_records)
            if len(oof_df) > 1_000_000:
                oof_csv_path = os.path.join(cfg.outdir, 'OOF_Predictions.csv')
                oof_df.to_csv(oof_csv_path, index=False)
                logging.info("OOF predictions too large for Excel, saved to: %s", oof_csv_path)
            else:
                oof_df.to_excel(writer, sheet_name='OOF_Predictions', index=False)
        
        logging.info("Training summary saved to: %s", summary_path)
        
    except Exception as e:
        logging.warning("Failed to save Excel summary: %s", str(e))
        # Fallback to CSV
        oof_csv_path = os.path.join(cfg.outdir, 'OOF_Predictions.csv')
        pd.DataFrame(oof_records).to_csv(oof_csv_path, index=False)
        logging.info("OOF predictions saved to CSV: %s", oof_csv_path)
    
    return model_path

def main():
    """Main training pipeline for LGBM ensemble."""
    args = parse_args()
    cfg = Config(args)
    
    # Setup logging
    log_file = setup_logging(cfg, "train_lgbm")
    
    # Ensure output directory exists
    ensure(cfg.outdir)
    
    # Validate configuration
    if not validate_config(cfg, required_files=['train_data'], script_type="training"):
        logging.error("Configuration validation failed. Exiting.")
        return 1
    
    # Set random seed for reproducibility
    set_seed(cfg.seed)
    
    try:
        # Load and preprocess training data
        hitgen_df = load_and_preprocess_training_data(cfg)
        
        # Create cluster groups for cross-validation
        hitgen_df = create_cluster_groups(hitgen_df, cfg)
        
        # Prepare balanced dataset with CV folds
        train_df, folds = prepare_balanced_dataset(hitgen_df, cfg)
          # Train LGBM ensemble
        lgbm, y_true = train_lgbm_ensemble(train_df, folds, cfg)
        
        # Save results
        model_path = save_training_results(lgbm, y_true, cfg)
        
        logging.info("LGBM training pipeline completed successfully")
        logging.info("Model saved to: %s", model_path)
        return 0
        
    except Exception as e:
        logging.error("Exception during LGBM training pipeline: %s", str(e), exc_info=True)
        try:
            error_path = os.path.join(cfg.outdir, "training_error_lgbm.log")
            with open(error_path, "w", encoding="utf-8") as f:
                import traceback
                f.write(traceback.format_exc())
            logging.info("Error traceback written to: %s", error_path)
        except Exception as e2:
            logging.error("Failed to write error traceback: %s", str(e2))
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
