import argparse
import torch

class Config:
    def __init__(self, args):
        self.train_data = args.train_data
        self.submission_template = getattr(args, 'submission_template', None)
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

def get_branch_hidden_dims(preset):
    if preset.lower() == 'deeper':
        return [2048, 1024]
    elif preset.lower() == 'wider':
        return [2048, 2048]
    elif preset.lower() == 'shallower':
        return [512, 256]
    else:
        return [1024, 512]

def parse_args():
    parser = argparse.ArgumentParser(description='DREAM Challenge WDR91 Training Script')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data file (e.g., data/WDR91_HitGen.parquet)')
    parser.add_argument('--submission-template', type=str, help='Path to submission template CSV file (required for prediction scripts)')
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
