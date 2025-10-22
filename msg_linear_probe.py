import argparse
import traceback
from datetime import datetime
from hsi_patch_loader import build_finetune_loader
from msg_utils import *
from dataset_configs import DATASETS
from get_model import get_model


def parse_args():
    parser = argparse.ArgumentParser("HSI Linear Probe")

    # ----- train/test
    parser.add_argument('--is_train', default=1, type=int)
    parser.add_argument('--is_test', default=1, type=int)
    parser.add_argument('--runs', default=5, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--val_interval', default=3, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--patience', default=10, type=int)

    # ----- optimization
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--keep_weights', default=1, type=int)  # keep top-k checkpoints
    parser.add_argument('--train_ratio', default=0.7, type=float)
    parser.add_argument('--val_ratio', default=0.15, type=float)

    # ----- paths
    parser.add_argument('--save_path', default='lp_results', type=str)
    parser.add_argument('--weights_path', default=None, type=str)

    # ----- model config
    parser.add_argument('--model', default='MSG', type=str)
    parser.add_argument('--mode', default='linear_probe', type=str)
    parser.add_argument('--channel_num', default=128, type=int)
    parser.add_argument('--crop_size', default=7, type=int)
    parser.add_argument('--patch_size', default=1, type=int)
    parser.add_argument('--encoder_depth', default=6, type=int)
    parser.add_argument('--decoder_depth', default=4, type=int)
    parser.add_argument('--head', default=8, type=int)
    parser.add_argument('--encoder_dim', default=128, type=int)
    parser.add_argument('--decoder_dim', default=64, type=int)
    parser.add_argument('--mask_ratio', default=0.7, type=float)
    parser.add_argument('--group_size', default=16, type=int)
    parser.add_argument('--mlp_ratio', default=4.0, type=float)
    parser.add_argument('--mask_ratio_contrast', default=0.3, type=float)
    parser.add_argument('--contrastive_lambda', default=1.0, type=float)
    parser.add_argument('--contrastive_temp', default=0.1, type=float)

    return parser.parse_args()


def run_one_dataset(args, dataset_name):
    """Run linear probe on a single dataset (train/val/test pipeline)."""

    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found in dataset_configs.py")

    # Dataset config
    cfg = DATASETS[dataset_name]
    args.dataset = dataset_name
    args.finetune_dataset = cfg["hsi_path"]
    args.finetune_gt = cfg["gt_path"]
    args.hsi_key = cfg["hsi_key"]
    args.gt_key = cfg["gt_key"]
    args.num_classes = cfg["num_classes"]

    print(f"\n\n=== Running dataset: {dataset_name} ===")
    os.makedirs(args.save_path, exist_ok=True)
    main_log_path = os.path.join(args.save_path, "main_log.txt")

    # Main log header
    with open(main_log_path, "a") as f:
        f.write(f"\n\n=== Started Linear Probe at {datetime.now()} ===\n")
        f.write(f"Command: {' '.join(sys.argv)}\n")
        f.write(f"PID: {os.getpid()}\n")

    oa_list = []
    df_list = []

    # Multi-run with different seeds
    for run_seed in range(1, args.runs + 1):
        print(f"\n======== Run {run_seed}/{args.runs} ========")
        with open(main_log_path, "a") as f:
            f.write(f"Starting seed {run_seed} at {datetime.now()}\n")

        # Reproducibility
        set_seed(run_seed)

        # Device
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Dataloaders
        train_loader, val_loader, test_loader = build_finetune_loader(
            hsi_path=args.finetune_dataset,
            gt_path=args.finetune_gt,
            hsi_key=args.hsi_key,
            gt_key=args.gt_key,
            patch_size=args.crop_size,
            batch_size=args.batch_size,
            stride=args.crop_size,
            shuffle=True,
            normalize=True,
            applyPCA=True,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=run_seed
        )

        # Model
        model = get_model(args).to(device)
        model_name = model.__class__.__name__

        # Output directory per run
        run_dir = os.path.join(args.save_path, model_name, args.dataset, f"seed_{run_seed}")
        os.makedirs(run_dir, exist_ok=True)

        # Train (if enabled)
        if args.is_train:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epoch - args.warmup_epochs)
            )
            scaler = torch.cuda.amp.GradScaler()
            criterion = torch.nn.CrossEntropyLoss(weight=compute_class_weights(train_loader, device))

            best_val_oa = 0.0
            epochs_no_improve = 0
            saved_models = []

            for epoch in range(args.epoch):
                # Warmup LR, then cosine schedule
                if epoch < args.warmup_epochs:
                    lr_scale = min(1.0, (epoch + 1) / args.warmup_epochs)
                    for pg in optimizer.param_groups:
                        pg['lr'] = args.lr * lr_scale

                # One training epoch
                train_loss, train_acc = Train(model, train_loader, optimizer, scaler, criterion, device, epoch)
                log_train_metrics(epoch, train_loss, train_acc, model_name, args.dataset, run_dir)

                if epoch >= args.warmup_epochs:
                    scheduler.step()

                # Periodic validation
                if (epoch + 1) % args.val_interval == 0:
                    val_OA, val_AA, val_Kappa, *_ = evaluate_model(model, val_loader, device, mode='Val')
                    print(f"[Val] Epoch {epoch:03d} | OA={val_OA:.5f} | AA={val_AA:.5f} | Kappa={val_Kappa:.5f}")

                    # Track best model by OA
                    if val_OA > best_val_oa:
                        best_val_oa = val_OA
                        epochs_no_improve = 0
                        ckpt = {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'val_OA': val_OA,
                            'args': vars(args)
                        }
                        ckpt_path = os.path.join(run_dir, f"best_seed{run_seed}_epoch{epoch}.pth")
                        torch.save(ckpt, ckpt_path)
                        saved_models.append((val_OA, epoch, ckpt_path))
                        saved_models.sort(key=lambda x: x[0], reverse=True)

                        # Keep top-k only
                        while len(saved_models) > args.keep_weights:
                            _, _, old_path = saved_models.pop()
                            if os.path.exists(old_path):
                                os.remove(old_path)
                    else:
                        epochs_no_improve += 1

                    # Early stopping
                    if epochs_no_improve >= args.patience:
                        print("Early stopping triggered.")
                        break

            # Select model for testing
            test_model_path = saved_models[0][2] if saved_models else args.weights_path
        else:
            test_model_path = args.weights_path

        # Test (if enabled)
        if args.is_test:
            checkpoint = torch.load(test_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=False)
            model.eval()

            OA, AA, Kappa, csv_path = test_model_and_save(
                args.dataset, model, test_loader, device, run_seed, run_dir
            )
            print(f"[Test] Seed {run_seed} | OA={OA:.6f} | AA={AA:.6f} | Kappa={Kappa:.6f}")
            oa_list.append(OA * 100)

            # Collect per-class metrics for summary
            df = pd.read_csv(csv_path, index_col=0)
            df_list.append(df)

        with open(main_log_path, "a") as f:
            f.write(f"Completed seed {run_seed} at {datetime.now()}\n")

    # Final summary across runs
    if oa_list:
        oa_mean = np.mean(oa_list)
        oa_std = np.std(oa_list)
        print(f"\n===== Final OA over {len(oa_list)} runs: {oa_mean:.2f} ± {oa_std:.2f} =====")

        # Remove 'support' if present
        df_list = [df.drop(columns=["support"], errors="ignore") for df in df_list]

        # Aggregate mean ± std per class/metric
        idx = df_list[0].index
        cols = df_list[0].columns
        summary_df = pd.DataFrame(index=idx, columns=cols)

        for col in cols:
            arr = np.stack([df[col].values for df in df_list], axis=0)
            mean = arr.mean(axis=0) * 100
            std = arr.std(axis=0) * 100
            summary_df[col] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(mean, std)]

        out_dir = os.path.join(args.save_path, model_name, args.dataset)
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, f"summary_{args.dataset}.csv")
        summary_df.to_csv(summary_path)
        print(f"Summary CSV saved to: {summary_path}")


def main():
    args = parse_args()

    # Datasets to evaluate
    datasets = ["Salinas", "Berlin", "WHU_Hi_LongKou"]

    for dataset_name in datasets:
        try:
            run_one_dataset(args, dataset_name)
        except Exception as e:
            print(f"Dataset {dataset_name} failed: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"Main process failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open("main_error_log.txt", "w") as f:
            f.write(error_msg)
        sys.exit(1)