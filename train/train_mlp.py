import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.cluster import MiniBatchKMeans
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import get_branch_hidden_dims
from utils.utils import (
    FP_LIST, FP_DIMS, ensure, set_seed, binarize_fp_string, 
    tensorize_fp, fp_string_to_tensor, parse_ecfp4_binary,
    setup_logging, validate_config
)
from models.multibranch_mlp import MultiBranchMLP

class FingerprintDataset(torch.utils.data.Dataset):
    """
    Dataset class for handling fingerprint data and labels from a DataFrame.
    Converts fingerprint strings into tensors for training or evaluation.
    
    Args:
        dataframe: Input dataframe containing fingerprint columns and labels
        fp_list: List of fingerprint names to process
        label_col: Name of the label column (default: "LABEL")
        device: PyTorch device for tensor placement
    """
    def __init__(self, dataframe, fp_list, label_col="LABEL", device=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.fp_list = fp_list
        self.label_col = label_col
        self.labels = torch.tensor(self.dataframe[self.label_col].values, dtype=torch.float32)
        self.fp_data = {}
        
        for fp_name in self.fp_list:
            if fp_name in self.dataframe.columns:
                self.fp_data[fp_name] = fp_string_to_tensor(
                    self.dataframe[fp_name], fp_name, device=device
                )
            else:
                fp_dim = FP_DIMS.get(fp_name, 2048)
                self.fp_data[fp_name] = tensorize_fp(
                    np.zeros((len(self.dataframe), fp_dim), dtype=np.uint8), device=device
                )
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        item_fps = {fp_name: data[idx] for fp_name, data in self.fp_data.items()}
        return item_fps, self.labels[idx].unsqueeze(-1)

def create_dataloader(df, fp_list, batch_size, shuffle, num_workers, 
                     label_col="LABEL", seed=42):
    """
    Create a DataLoader for the FingerprintDataset with batching and parallel loading.
    
    Args:
        df: Input dataframe
        fp_list: List of fingerprint names
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of parallel workers
        label_col: Name of the label column
        seed: Random seed for reproducibility
        
    Returns:
        DataLoader: Configured PyTorch DataLoader
    """
    dataset = FingerprintDataset(df, fp_list, label_col=label_col)
    actual_batch_size = min(batch_size, len(df))
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    return DataLoader(
        dataset, 
        batch_size=actual_batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=True, 
        generator=g, 
        drop_last=False
    )

def train_epoch(model, dataloader, optimizer, criterion, device, epoch_num):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model to train
        dataloader: Training data loader
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: PyTorch device
        epoch_num: Current epoch number for logging
        
    Returns:        tuple: (average_loss, true_labels, predicted_probabilities)
    """
    
    model.train()
    total_loss = 0
    all_preds_proba = []
    all_true_labels = []
    
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch_num+1} Train", leave=False)
    
    for batch_data, batch_labels in progress_bar:
        batch_fps = {fp_name: data.to(device) for fp_name, data in batch_data.items()}
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_fps)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds_proba.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())
        all_true_labels.extend(batch_labels.detach().cpu().numpy().flatten())
        progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_true_labels), np.array(all_preds_proba)

def evaluate_epoch(model, dataloader, criterion, device, epoch_num):
    """
    Evaluate the model for one epoch.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: Validation data loader
        criterion: Loss function
        device: PyTorch device
        epoch_num: Current epoch number for logging
        
    Returns:        tuple: (average_loss, true_labels, predicted_probabilities)
    """
    
    model.eval()
    total_loss = 0
    all_preds_proba = []
    all_true_labels = []
    
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch_num+1} Val", leave=False)
    
    with torch.no_grad():
        for batch_data, batch_labels in progress_bar:
            batch_fps = {fp_name: data.to(device) for fp_name, data in batch_data.items()}
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_fps)
            loss = criterion(outputs, batch_labels)
            
            total_loss += loss.item()
            all_preds_proba.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())
            all_true_labels.extend(batch_labels.detach().cpu().numpy().flatten())
            progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_true_labels), np.array(all_preds_proba)

def calculate_metrics(y_true, y_pred_proba, y_pred_binary):
    """
    Calculate comprehensive evaluation metrics for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        y_pred_binary: Predicted binary labels
        
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    metrics = {}
    
    try:
        metrics["AUC"] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics["AUC"] = 0.0
    
    try:
        metrics["AUPRC"] = average_precision_score(y_true, y_pred_proba)
    except ValueError:
        metrics["AUPRC"] = 0.0
    
    metrics["F1"] = f1_score(y_true, y_pred_binary, zero_division=0)
    metrics["Precision"] = precision_score(y_true, y_pred_binary, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred_binary, zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        metrics["TP"] = int(tp)
        metrics["FP"] = int(fp)
        metrics["TN"] = int(tn)
        metrics["FN"] = int(fn)
    except ValueError:
        metrics["TP"] = int(np.sum((y_true == 1) & (y_pred_binary == 1)))
        metrics["FP"] = int(np.sum((y_true == 0) & (y_pred_binary == 1)))
        metrics["TN"] = int(np.sum((y_true == 0) & (y_pred_binary == 0)))
        metrics["FN"] = int(np.sum((y_true == 1) & (y_pred_binary == 0)))
    
    return metrics

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

def train_single_fold(fold_idx, train_indices, val_indices, train_df, cfg):
    """
    Train a single fold of the MLP model.
    
    Args:
        fold_idx: Current fold index
        train_indices: Training sample indices for this fold
        val_indices: Validation sample indices for this fold
        train_df: Complete training dataframe
        cfg: Configuration object
        
    Returns:
        str: Path to saved best model for this fold
    """
    logging.info("---------- Fold %d/%d ----------", fold_idx+1, cfg.n_splits)
    
    df_train_fold = train_df.iloc[train_indices]
    df_val_fold = train_df.iloc[val_indices]
    
    # Calculate positive class weight for imbalanced data
    n_pos = int((df_train_fold['LABEL'] == 1).sum())
    n_neg = int((df_train_fold['LABEL'] == 0).sum())
    pos_weight = n_neg / max(1, n_pos) * cfg.positive_class_weight
    logging.info("Fold %d pos_weight=%.2f", fold_idx+1, pos_weight)
    
    # Create data loaders
    train_loader = create_dataloader(
        df_train_fold, FP_LIST, cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, seed=cfg.seed+fold_idx
    )
    val_loader = create_dataloader(
        df_val_fold, FP_LIST, cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, seed=cfg.seed+fold_idx
    )
    
    # Initialize model
    model = MultiBranchMLP(
        fp_input_dims=FP_DIMS,
        branch_hidden_dims=cfg.branch_hidden_dims,
        branch_embedding_dim=cfg.branch_embedding_dim,
        common_hidden_dims=cfg.common_hidden_dims,
        dropout_rate=cfg.dropout_rate,
        fp_list=FP_LIST
    ).to(cfg.device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.l2_decay)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=cfg.device)
    )
    
    # Training loop with early stopping
    best_auprc = -1.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(cfg.epochs):
        train_loss, _, _ = train_epoch(model, train_loader, optimizer, criterion, cfg.device, epoch)
        val_loss, val_true, val_pred_proba = evaluate_epoch(model, val_loader, criterion, cfg.device, epoch)
        
        val_true = np.array(val_true).astype(int)
        val_pred_proba = np.array(val_pred_proba)
        val_pred_binary = (val_pred_proba >= 0.5).astype(int)
        
        metrics = calculate_metrics(val_true, val_pred_proba, val_pred_binary)
        
        logging.info(
            "Fold %d Epoch %d | Loss: %.4f | AUC: %.4f | AUPRC: %.4f | F1: %.4f | "
            "Precision: %.4f | Recall: %.4f | TP: %d | FP: %d | TN: %d | FN: %d",
            fold_idx+1, epoch+1, val_loss, metrics['AUC'], metrics['AUPRC'], 
            metrics['F1'], metrics['Precision'], metrics['Recall'],
            metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN']
        )
        
        # Early stopping logic
        if metrics['AUPRC'] > best_auprc + 1e-4:
            best_auprc = metrics['AUPRC']
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= cfg.early_stopping_patience:
            logging.info("Early stopping triggered after %d epochs without improvement in AUPRC", 
                        cfg.early_stopping_patience)
            break
    
    # Save best model for this fold
    if best_model_state is not None:
        model_save_path = os.path.join(cfg.outdir, f"best_model_fold{fold_idx+1}.pt")
        torch.save(best_model_state, model_save_path)
        logging.info("Saved best model for fold %d to: %s", fold_idx+1, model_save_path)
        return model_save_path
    else:
        logging.warning("No best model state found for fold %d", fold_idx+1)
        return None

def collect_oof_predictions(folds, train_df, cfg):
    """
    Collect out-of-fold predictions from all trained models.
    
    Args:
        folds: List of (train_indices, val_indices) tuples
        train_df: Complete training dataframe
        cfg: Configuration object
        
    Returns:
        tuple: (fold_metrics, oof_records) - metrics and predictions for all folds
    """
    logging.info("Collecting out-of-fold predictions...")
    
    fold_metrics = []
    oof_records = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        df_val_fold = train_df.iloc[val_indices]
        
        # Load best model for this fold
        model_save_path = os.path.join(cfg.outdir, f"best_model_fold{fold_idx+1}.pt")
        
        if not os.path.exists(model_save_path):
            logging.warning("Model for fold %d not found: %s", fold_idx+1, model_save_path)
            continue
        
        try:
            model = MultiBranchMLP(
                fp_input_dims=FP_DIMS,
                branch_hidden_dims=cfg.branch_hidden_dims,
                branch_embedding_dim=cfg.branch_embedding_dim,
                common_hidden_dims=cfg.common_hidden_dims,
                dropout_rate=cfg.dropout_rate,
                fp_list=FP_LIST
            ).to(cfg.device)
            
            model.load_state_dict(torch.load(model_save_path, map_location=cfg.device))
            
            val_loader = create_dataloader(
                df_val_fold, FP_LIST, cfg.batch_size, shuffle=False, 
                num_workers=cfg.num_workers, seed=cfg.seed+fold_idx
            )
            
            _, val_true, val_pred_proba = evaluate_epoch(
                model, val_loader, torch.nn.BCEWithLogitsLoss(), cfg.device, 0
            )
            
            val_true = np.array(val_true).astype(int)
            val_pred_proba = np.array(val_pred_proba)
            val_pred_binary = (val_pred_proba >= 0.8).astype(int)
            
            metrics = calculate_metrics(val_true, val_pred_proba, val_pred_binary)
            metrics.update({
                'fold': fold_idx+1,
                'n_val': len(val_true),
                'n_pos': int(val_true.sum()),
                'n_neg': int((val_true==0).sum())
            })
            fold_metrics.append(metrics)
            
            # OOF records
            for i, idx in enumerate(val_indices):
                oof_records.append({
                    'index': idx,
                    'compound_id': (df_val_fold.index[i] if 'compound_id' not in df_val_fold.columns 
                                  else df_val_fold.iloc[i]['compound_id']),
                    'true_label': val_true[i],
                    'pred_proba': val_pred_proba[i],
                    'fold': fold_idx+1
                })
                
            logging.info("Fold %d OOF predictions collected successfully", fold_idx+1)
            
        except Exception as e:
            logging.error("Failed to collect OOF predictions for fold %d: %s", fold_idx+1, str(e))
    
    return fold_metrics, oof_records

def save_training_results(fold_metrics, oof_records, cfg):
    """
    Save training results and summary to Excel file.
    
    Args:
        fold_metrics: List of metrics dictionaries for each fold
        oof_records: List of out-of-fold prediction records
        cfg: Configuration object
        
    Returns:
        str: Path to saved summary file
    """
    logging.info("Saving training results...")
    
    summary_path = os.path.join(cfg.outdir, 'training_summary.xlsx')
    
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
            pd.DataFrame(oof_records).to_excel(
                writer, sheet_name='OOF_Predictions', index=False
            )
        
        logging.info("Training summary saved to: %s", summary_path)
        
    except Exception as e:
        logging.warning("Failed to save Excel summary: %s", str(e))
        # Fallback to CSV
        oof_csv_path = os.path.join(cfg.outdir, 'OOF_Predictions.csv')
        pd.DataFrame(oof_records).to_csv(oof_csv_path, index=False)
        logging.info("OOF predictions saved to CSV: %s", oof_csv_path)
    
    return summary_path

def parse_mlp_training_args():
    """Parse command line arguments for MLP training script."""
    parser = argparse.ArgumentParser(description='DREAM Challenge WDR91 MLP Training Script')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data file (e.g., toy_data/toy_train_data.parquet)')
    parser.add_argument('--outdir', type=str, default='./output/', help='Output directory for results and models (default: ./output/)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers (default: 4)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu). Default: auto-detect')
    parser.add_argument('--positive-class-weight', type=float, default=1.0, help='Weight for positive class in BCEWithLogitsLoss')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate for optimizer')
    parser.add_argument('--l2-decay', type=float, default=3e-5, help='L2 weight decay for optimizer')
    parser.add_argument('--branch-hidden-dims', type=str, default='deeper', choices=['deeper', 'wider', 'shallower'], help='Branch hidden dimension preset: deeper, wider, shallower')
    parser.add_argument('--branch-embedding-dim', type=int, default=256, help='Output embedding size for each FP branch')
    parser.add_argument('--common-hidden-dims', type=int, nargs=2, default=[1024, 512], help='Hidden layer sizes for common MLP part (two integers)')
    parser.add_argument('--dropout-rate', type=float, default=0.4, help='Dropout rate applied in all MLP layers')
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of CV splits')
    parser.add_argument('--use-scaffold-split', action='store_true', help='Use scaffold split (not implemented)')
    parser.add_argument('--grouping-strategy', type=str, default='auto', choices=['auto', 'bb', 'kmeans'], help='Grouping strategy for cluster-aware CV')
    return parser.parse_args()

class MLPTrainingConfig:
    """Simple configuration class for MLP training scripts."""
    def __init__(self, args):
        self.train_data = args.train_data
        self.submission_template = None  # Not needed for training
        self.outdir = args.outdir
        self.seed = args.seed
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.branch_hidden_dims = get_branch_hidden_dims(args.branch_hidden_dims)
        self.branch_embedding_dim = args.branch_embedding_dim
        self.common_hidden_dims = args.common_hidden_dims
        self.dropout_rate = args.dropout_rate
        self.l2_decay = args.l2_decay
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.positive_class_weight = args.positive_class_weight
        self.early_stopping_patience = args.early_stopping_patience
        self.num_workers = args.workers
        self.n_splits = args.n_splits
        self.use_scaffold_split = args.use_scaffold_split
        self.grouping_strategy = args.grouping_strategy

def main():
    """Main training pipeline for MLP ensemble."""
    args = parse_mlp_training_args()
    cfg = MLPTrainingConfig(args)
    
    # Setup logging
    log_file = setup_logging(cfg, "train_mlp")
    
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
        
        # Train all folds
        model_paths = []
        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            model_path = train_single_fold(fold_idx, train_indices, val_indices, train_df, cfg)
            if model_path:
                model_paths.append(model_path)
        
        if not model_paths:
            raise RuntimeError("No models were successfully trained")
        
        # Collect out-of-fold predictions
        fold_metrics, oof_records = collect_oof_predictions(folds, train_df, cfg)
        
        # Save results
        summary_path = save_training_results(fold_metrics, oof_records, cfg)
        
        logging.info("MLP training pipeline completed successfully")
        logging.info("Trained %d models: %s", len(model_paths), model_paths)
        logging.info("Summary saved to: %s", summary_path)
        return 0
        
    except Exception as e:
        logging.error("Exception during MLP training pipeline: %s", str(e), exc_info=True)
        try:
            error_path = os.path.join(cfg.outdir, "training_error_mlp.log")
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
