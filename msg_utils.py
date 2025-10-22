import os
import sys
import random
import gc
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


def Pretrain(dataloader, model, optimizer, scaler, epoch, device):
    """
    Generic pretraining loop for one epoch.
    """
    model.train()
    total_loss = 0.0
    loader_length = len(dataloader)

    use_amp = (device.type == 'cuda')

    for batch_idx, hsi in enumerate(dataloader):
        hsi = hsi.to(device)

        with torch.autocast(device_type='cuda', enabled=use_amp):
            loss, _ = model(hsi)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / loader_length
    print(f"[Pretrain] Epoch {epoch} | Loss={avg_loss:.4f}", flush=True)
    return avg_loss


def Train(model, dataloader, optimizer, scaler, criterion, device, epoch):
    """
    Supervised training loop for one epoch.
    """
    model.train()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    use_amp = (device.type == 'cuda')

    for i, (hsi, labels) in enumerate(dataloader):
        hsi = hsi.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type='cuda', enabled=use_amp):
            outputs = model(hsi)
            loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        del outputs, preds, hsi, labels

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / max(1, total_samples)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Train]  Epoch {epoch} | Loss={avg_loss:.5f} | OA={avg_acc:.5f}", flush=True)
    return avg_loss, avg_acc


def evaluate_model(model, dataloader, device, mode='Val', show_tqdm=True):
    """
    Evaluate model and return metrics, reports, and predictions.
    """
    model.eval()
    y_true, y_pred = [], []

    iterator = enumerate(dataloader)
    if show_tqdm:
        iterator = tqdm(iterator, total=len(dataloader), desc=f"{mode} Eval", file=sys.stdout)

    with torch.no_grad():
        for _, (hsi, labels) in iterator:
            hsi = hsi.to(device)
            labels = labels.to(device)
            outputs = model(hsi)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            del outputs, preds, hsi, labels

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    OA, AA, Kappa, ClassAccuracy = output_metric(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return OA, AA, Kappa, ClassAccuracy, report, cm, y_true, y_pred


def save_model(model, optimizer, epoch, loss, args):
    """
    Save pretraining checkpoint and return file path.
    """
    model_name = model.__class__.__name__
    file_name = f"{model_name}_{args.mode}_epoch{epoch}_loss{loss:.6f}.pth"
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)

    torch.save(
        {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss,
        },
        file_path,
    )
    return file_path


def save_model_ablation(model, optimizer, epoch, loss, args, hp):
    """
    Save checkpoint for ablation runs and return file path.
    """
    model_name = model.__class__.__name__
    file_name = f"{model_name}_{args.hp_name}_{hp}_epoch{epoch}_loss{loss:.6f}.pth"
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)

    torch.save(
        {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss,
        },
        file_path,
    )
    return file_path


def test_model_and_save(dataset, model, test_loader, device, seed, base_dir):
    """
    Test the model, write per-class/overall metrics CSV, and return headline metrics + CSV path.
    """
    OA, AA, Kappa, ClassAccuracy, report, cm, y_true, y_pred = evaluate_model(
        model, test_loader, device, mode='Test'
    )

    os.makedirs(base_dir, exist_ok=True)
    model_name = model.__class__.__name__
    mode = getattr(model, "mode", "unknown")
    prefix = f"{model_name}_{mode}_{dataset}"
    csv_path = os.path.join(base_dir, f"test_metrics_{prefix}_{seed}.csv")

    report_dict = classification_report(
        y_true, y_pred, output_dict=True, digits=6, zero_division=0
    )
    class_labels = [str(i) for i in sorted(np.unique(y_true))]

    # Per-class metrics table
    class_data = []
    for i, label in enumerate(class_labels):
        class_metrics = report_dict[label]
        class_metrics['accuracy'] = ClassAccuracy[i]
        class_data.append(class_metrics)
    df_classes = pd.DataFrame(class_data, index=class_labels)

    # Overall metrics table (OA/AA/Kappa)
    df_overall = pd.DataFrame(
        {
            'precision': [np.nan, np.nan, np.nan],
            'recall': [np.nan, np.nan, np.nan],
            'f1-score': [np.nan, np.nan, np.nan],
            'support': [np.nan, np.nan, np.nan],
            'accuracy': [OA, AA, Kappa],
        },
        index=['OA', 'AA', 'Kappa'],
    )

    final_df = pd.concat([df_classes, df_overall])
    final_df = final_df[['accuracy', 'precision', 'recall', 'f1-score', 'support']]
    final_df.to_csv(csv_path, index=True)

    print(f"[Test] Saved metrics CSV: {csv_path}", flush=True)
    return OA, AA, Kappa, csv_path


def output_metric(tar, pre):
    """
    Compute headline metrics from ground truth and predictions.
    """
    matrix = confusion_matrix(tar, pre)
    OA, AA, Kappa, ClassAccuracy = cal_results(matrix)
    return OA, AA, Kappa, ClassAccuracy


def cal_results(matrix):
    """
    Derive OA, AA, Kappa, and per-class accuracy from a confusion matrix.
    """
    shape = np.shape(matrix)
    number = 0
    sum = 0  # keep naming for backward compatibility
    ClassAccuracy = np.zeros([shape[0]], dtype=np.float64)

    for i in range(shape[0]):
        number += matrix[i, i]
        ClassAccuracy[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])

    OA = number / np.sum(matrix)
    AA = np.mean(ClassAccuracy)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA, Kappa, ClassAccuracy


def log_metrics(epoch, avg_loss, avg_loss_rec, avg_loss_con, lr, epoch_time, model_name):
    """
    Log pretraining losses and LR to console and a persistent text file.
    """
    log_str = (
        f"Epoch: {epoch} | Loss: {avg_loss:.6f} | "
        f"Rec: {avg_loss_rec:.6f} | Con: {avg_loss_con:.6f} | "
        f"LR: {lr:.2e} | Time: {epoch_time:.2f}s"
    )
    print(log_str, flush=True)

    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{model_name}_pretrain_metrics.txt")

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Epoch\tLoss\tLoss_Rec\tLoss_Con\tLR\tTime\n")

    with open(log_file, "a") as f:
        f.write(f"{epoch}\t{avg_loss:.6f}\t{avg_loss_rec:.6f}\t{avg_loss_con:.6f}\t{lr:.6f}\t{epoch_time:.2f}\n")


def log_train_metrics(epoch, train_loss, train_acc, model_name, dataset, base_dir):
    """
    Append per-epoch training metrics to a run-local text file.
    """
    os.makedirs(base_dir, exist_ok=True)
    log_file = os.path.join(base_dir, f"{model_name}_{dataset}_train_metrics.txt")
    with open(log_file, "a") as f:
        f.write(f"Epoch: {epoch}, Loss: {train_loss:.6f}, Accuracy: {train_acc:.6f}\n")


def compute_class_weights(train_loader, device):
    """
    Compute balanced class weights (sqrt of inverse frequency) for CE loss.
    """
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.cpu().numpy().tolist())
    raw_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    return torch.tensor(np.sqrt(raw_weights), dtype=torch.float, device=device)


def set_seed(seed):
    """
    Set random seeds for reproducibility (Python, NumPy, PyTorch).
    """
    # Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed