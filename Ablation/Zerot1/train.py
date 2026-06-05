import os
import warnings
import sys
warnings.filterwarnings('ignore')
import argparse
import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm

from AblateZerot1 import PredictModel
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root)
from AB_Data import AB_Data
from utils import score  

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class Config:

    CUDA_DEVICE = '1'
    DATA_FILE_PATH = os.path.join(ROOT, "data/data.xlsx")
    EMBEDDING_PATHS = [
        os.path.join(ROOT, "Embeddings/antibinder_heavy.pkl"),
        os.path.join(ROOT, "Embeddings/Light.pkl"),
        os.path.join(ROOT, "Embeddings/Antigen.pkl"),
    ]
    PRETRAINED_MODEL_PATH = os.path.join(ROOT, "model_weights/modified_model.pth")
    CHECKPOINT_DIR = "ckpts"

    RESULTS_CSV = "Ablation_t1zero.csv"
    LOG_DIR = "Ablatin_t1zero"
    PLOT_DIR = "training_curves_LP"

    NUM_LAYERS = 6
    D_MODEL = 256
    DFF = 512
    NUM_HEADS = 8
    VOCAB_SIZE = 18

    SEEDS = [1]
    BATCH_SIZE = 8
    MAX_EPOCHS = 100
    MIN_CHECKPOINT_EPOCH = 20
    PATIENCE = 30  # Early stopping patience

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    GRAD_CLIP_NORM = 1.0

    SCHEDULER_MODE = 'max'  # 'max' for metrics to maximize (PRAUC, AUC, Acc)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR = 1e-7

    USE_POS_WEIGHT = True  # Set to True to use class weights
    POS_WEIGHT_VALUE = 0.3  # Only used if USE_POS_WEIGHT is True

    PRIMARY_METRIC = 'auc'  # Metric to optimize ('prauc', 'auc', 'acc')

    VERBOSE = True
    SHOW_PROGRESS_BARS = True
    PROGRESS_BAR_WIDTH = 30

    PRINT_FULL_TRACEBACK = True

def set_seed(seed):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model, data_loader, device, criterion):
    """
    Evaluate model and return loss, unified metrics dict from utils.score, labels, and outputs
    """
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for x1, x1maccs, x2, x2maccs, t1, t2, t3, aac1, aac2, aac3, t4, labels in data_loader:
            x1, x2 = x1.to(device), x2.to(device)
            t1, t2, t3, t4 = t1.to(device), t2.to(device), t3.to(device), t4.to(device)
            x1maccs, x2maccs = x1maccs.to(device), x2maccs.to(device)
            aac1, aac2, aac3 = aac1.to(device), aac2.to(device), aac3.to(device)
            
            labels = labels.to(device)

            outputs = model(x1, x1maccs, x2, x2maccs, t1, t2, t3, aac1, aac2, aac3, t4, 
                            mask1=None, adjoin_matrix1=None, mask2=None, adjoin_matrix2=None)

            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            all_outputs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    
    metrics = score(all_labels, all_outputs)

    return avg_loss, metrics, all_labels, all_outputs


def train_one_seed(seed, config, device, UNIQUE_SPLIT):
    """
    Train with improved checkpointing strategy using unified score function.
    """
    set_seed(seed)

    os.makedirs(config.LOG_DIR, exist_ok=True)
    csv_log_path = os.path.join(config.LOG_DIR, f'training_log_seed_{seed}.csv')

    with open(csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'train_auc', 'train_prauc',
                         'val_loss', 'val_acc', 'val_auc', 'val_prauc',
                         'val_sens', 'val_spec', 'learning_rate', 'status'])

    if config.VERBOSE:
        print(f"[INFO] Seed {seed} | Loading data from: {config.DATA_FILE_PATH}")

    dataset = AB_Data(config.DATA_FILE_PATH, config.EMBEDDING_PATHS)
    if UNIQUE_SPLIT:
        print("****************************************************************\n\tUsing Unique Antigen-Antibody Pair Split\n****************************************************************")
        train_loader, val_loader, test_loader = dataset.get_dataloaders_from_csv(train_csv= os.path.join(ROOT, "data/train.csv"),val_csv= os.path.join(ROOT, "data/val.csv"),test_csv= os.path.join(ROOT, "data/test.csv"),batch_size=config.BATCH_SIZE)
    else:
        print("****************************************************************\n\t\tUsing Random Split\n****************************************************************")
        train_loader, val_loader, test_loader = dataset.get_dataloaders(
            batch_size=config.BATCH_SIZE,
            seed=seed
        )

    if config.VERBOSE:
        print(f"[INFO] Seed {seed} | Data loaded successfully")

    model = PredictModel(
        num_layers=config.NUM_LAYERS,
        d_model=config.D_MODEL,
        dff=config.DFF,
        num_heads=config.NUM_HEADS,
        vocab_size=config.VOCAB_SIZE
    ).to(device)

    if os.path.exists(config.PRETRAINED_MODEL_PATH):
        checkpoint = torch.load(config.PRETRAINED_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        if config.VERBOSE:
            print(f"[INFO] Seed {seed} | Loaded pretrained weights")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=config.SCHEDULER_MODE,
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR
    )

    if config.USE_POS_WEIGHT:
        pos_weight = torch.tensor([config.POS_WEIGHT_VALUE]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if config.VERBOSE:
            print(f"[INFO] Seed {seed} | Using pos_weight={pos_weight.item():.3f}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_metrics = {
        'pr_auc': 0.0,
        'roc_auc': 0.0,
        'accuracy': 0.0,
        'loss': 999.0,
        'sensitivity': 0.0,
        'specificity': 0.0
    }
    
    best_epoch = -1
    stopping_monitor = 0

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'ADC_{seed}_best_model.pth')

    if config.SHOW_PROGRESS_BARS:
        epoch_pbar = tqdm(range(config.MAX_EPOCHS), desc=f'Seed {seed}',
                          bar_format='{l_bar}{bar:' + str(config.PROGRESS_BAR_WIDTH) + '}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                          ncols=100)
    else:
        epoch_pbar = range(config.MAX_EPOCHS)

    for epoch in epoch_pbar:
        model.train()
        train_loss_sum = 0
        train_outputs = []
        train_labels = []

        if config.SHOW_PROGRESS_BARS:
            batch_pbar = tqdm(train_loader, desc=f'  Epoch {epoch}', leave=False,
                              bar_format='{l_bar}{bar:20}| {n_fmt}/{total_fmt}',
                              ncols=90)
        else:
            batch_pbar = train_loader
        for x1, x1maccs, x2, x2maccs, t1, t2, t3, aac1, aac2, aac3, t4, labels in batch_pbar:
            x1, x2 = x1.to(device), x2.to(device)
            t1, t2, t3, t4 = t1.to(device), t2.to(device), t3.to(device), t4.to(device)
            x1maccs, x2maccs = x1maccs.to(device), x2maccs.to(device)
            aac1, aac2, aac3 = aac1.to(device), aac2.to(device), aac3.to(device)
            
            labels = labels.to(device)

            outputs = model(x1, x1maccs, x2, x2maccs, t1, t2, t3, aac1, aac2, aac3, t4,
                            mask1=None, adjoin_matrix1=None, mask2=None, adjoin_matrix2=None)

            loss = criterion(outputs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
            optimizer.step()

            train_loss_sum += loss.item()
            
            probs = torch.sigmoid(outputs)
            train_outputs.extend(probs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            if config.SHOW_PROGRESS_BARS and hasattr(batch_pbar, 'set_postfix'):
                batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss = train_loss_sum / len(train_loader)
        train_metrics = score(train_labels, train_outputs)

        val_loss, val_metrics, _, _ = evaluate_model(
            model, val_loader, device, criterion
        )

        if config.PRIMARY_METRIC == 'prauc':
            primary_val = val_metrics['pr_auc']
        elif config.PRIMARY_METRIC == 'auc':
            primary_val = val_metrics['roc_auc']
        elif config.PRIMARY_METRIC == 'acc':
            primary_val = val_metrics['accuracy']
        else:
            primary_val = val_metrics['pr_auc']

        scheduler.step(primary_val)
        current_lr = optimizer.param_groups[0]['lr']

        status = ''
        if epoch >= config.MIN_CHECKPOINT_EPOCH:
            is_better = False
            
            curr_prauc = val_metrics['pr_auc']
            curr_auc = val_metrics['roc_auc']
            curr_acc = val_metrics['accuracy']
            
            best_prauc = best_val_metrics['pr_auc']
            best_auc = best_val_metrics['roc_auc']
            best_acc = best_val_metrics['accuracy']

            if config.PRIMARY_METRIC == 'prauc':
                is_better = (curr_prauc > best_prauc or
                             (curr_prauc == best_prauc and curr_acc > best_acc))
            elif config.PRIMARY_METRIC == 'auc':
                is_better = (curr_auc > best_auc or
                             (curr_auc == best_auc and curr_acc > best_acc))
            elif config.PRIMARY_METRIC == 'acc':
                is_better = (curr_acc > best_acc or
                             (curr_acc == best_acc and curr_prauc > best_prauc))

            if is_better:
                best_val_metrics = val_metrics.copy()
                best_val_metrics['loss'] = val_loss
                best_epoch = epoch
                stopping_monitor = 0
                torch.save(model.state_dict(), checkpoint_path)
                status = 'BEST'
                if config.VERBOSE:
                    tqdm.write(f'Epoch {epoch:3d} | New best! Val AUC: {curr_auc:.4f}, Val Acc: {curr_acc:.4f}')
            else:
                stopping_monitor += 1
                status = 'NO_IMPROVE'
        else:
            status = 'WARMUP'

        with open(csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, 
                             f'{train_loss:.6f}', f"{train_metrics['accuracy']:.4f}",
                             f"{train_metrics['roc_auc']:.4f}", f"{train_metrics['pr_auc']:.4f}",
                             f'{val_loss:.6f}', f"{val_metrics['accuracy']:.4f}",
                             f"{val_metrics['roc_auc']:.4f}", f"{val_metrics['pr_auc']:.4f}",
                             f"{val_metrics['sensitivity']:.4f}", f"{val_metrics['specificity']:.4f}",
                             f'{current_lr:.2e}', status])

        if config.SHOW_PROGRESS_BARS and hasattr(epoch_pbar, 'set_postfix'):
            epoch_pbar.set_postfix({'val_auc': f"{val_metrics['roc_auc']:.3f}", 
                                    'val_spec': f"{val_metrics['specificity']:.3f}"})

        if stopping_monitor >= config.PATIENCE:
            if config.VERBOSE:
                print(f"\n[INFO] Early stop at epoch {epoch}. Best Val AUC: {best_val_metrics['roc_auc']:.4f} at epoch {best_epoch}.")
            break

    if best_epoch == -1:
        print(f"\n[WARNING] Seed {seed}: No checkpoint saved.")
        return None

    if config.VERBOSE:
        print(f"\n[EVAL] Seed {seed} | Best Epoch: {best_epoch} | Val AUC: {best_val_metrics['roc_auc']:.4f}")

    model.load_state_dict(torch.load(checkpoint_path))

    test_loss, test_metrics, test_labels, test_outputs = evaluate_model(
        model, test_loader, device, criterion
    )

    results = {
        'seed': seed,
        'best_epoch': best_epoch,
        'val_loss': best_val_metrics['loss'],
        'val_acc': best_val_metrics['accuracy'],
        'val_auc': best_val_metrics['roc_auc'],
        'val_prauc': best_val_metrics['pr_auc'],
        'val_sensitivity': best_val_metrics['sensitivity'],
        'val_specificity': best_val_metrics['specificity'],
        'test_loss': test_loss,
        'test_se': test_metrics['sensitivity'],
        'test_sp': test_metrics['specificity'],
        'test_mcc': test_metrics['mcc'],
        'test_acc': test_metrics['accuracy'],
        'test_auc': test_metrics['roc_auc'],
        'test_f1': test_metrics['f1_score'],
        'test_ba': test_metrics['balanced_accuracy'],
        'test_prauc': test_metrics['pr_auc'],
        'test_ppv': test_metrics['ppv'],
        'test_npv': test_metrics['npv'],
    }

    if config.VERBOSE:
        print(f"[RESULT] Seed {seed} | Test PRAUC: {results['test_prauc']:.4f} | Test AUC: {results['test_auc']:.4f} | Test Acc: {results['test_acc']:.4f}")

    return results



def main(unique_split):
    UNIQUE_SPLIT = unique_split
    os.environ['CUDA_VISIBLE_DEVICES'] = Config.CUDA_DEVICE

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")

    with open(Config.RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'seed', 'best_epoch',
            'val_loss', 'val_acc', 'val_auc', 'val_prauc', 'val_sensitivity', 'val_specificity',
            'test_loss', 'test_se', 'test_sp', 'test_mcc', 'test_acc', 'test_auc', 
            'test_f1', 'test_ba', 'test_prauc', 'test_ppv', 'test_npv'
        ])

    print(f"\n{'='*80}")
    print(f"Results will be saved to: {Config.RESULTS_CSV}")
    print(f"Running seeds: {Config.SEEDS[0]}-{Config.SEEDS[-1]}")
    print(f"{'='*80}\n")

    all_results = []
    for seed in Config.SEEDS:
        print(f"\n{'#'*80}")
        print(f"# SEED {seed}")
        print(f"{'#'*80}")

        result = train_one_seed(seed, Config, device, UNIQUE_SPLIT)

        if result is not None:
            all_results.append(result)

            with open(Config.RESULTS_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result['seed'],
                    result['best_epoch'],
                    f"{result['val_loss']:.6f}",
                    f"{result['val_acc']:.4f}",
                    f"{result['val_auc']:.4f}",
                    f"{result['val_prauc']:.4f}",
                    f"{result['val_sensitivity']:.4f}",
                    f"{result['val_specificity']:.4f}",
                    f"{result['test_loss']:.6f}",
                    f"{result['test_se']:.4f}",
                    f"{result['test_sp']:.4f}",
                    f"{result['test_mcc']:.4f}",
                    f"{result['test_acc']:.4f}",
                    f"{result['test_auc']:.4f}",
                    f"{result['test_f1']:.4f}",
                    f"{result['test_ba']:.4f}",
                    f"{result['test_prauc']:.4f}",
                    f"{result['test_ppv']:.4f}",
                    f"{result['test_npv']:.4f}"
                ])

    print(f"\n{'='*80}")
    print(f"ALL SEEDS COMPLETED ({len(all_results)}/{len(Config.SEEDS)} successful)")
    print(f"{'='*80}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ABFormer trainer")
    parser.add_argument("--unique_split", action='store_true', help="Random Split Training / Unique Split Training")
    args = parser.parse_args()
    main(args.unique_split)
