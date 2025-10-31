import time
import argparse
from hsi_patch_loader import build_multi_pretrain_loader_npy
from msg_utils import *
from get_model import get_model


def parse_args():
    parser = argparse.ArgumentParser("HSI Pretrain")

    # ----- training setup
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--pretrain_dataset', default='/data/hyspecnet-GlobalGWPCA', type=str)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--pretrain_num', default=100000, type=int)
    parser.add_argument('--keep_weights', default=1, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--save_dir', default='weights', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--weights_path', type=str, default=None)

    # ----- model config
    parser.add_argument('--model', default='MSG', type=str)
    parser.add_argument('--mode', default='pretrain', type=str)
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
    parser.add_argument('--num_classes', default=100, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Single-seed loop (kept as list for easy extension)
    seeds = [1]
    for seed in seeds:
        args.seed = seed
        set_seed(seed)

        # Build pretraining data loader
        pretrain_loader = build_multi_pretrain_loader_npy(
            base_folder=args.pretrain_dataset,
            npy_pattern='*.npy',
            sample_size=args.crop_size,
            stride=args.crop_size,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            print_total=True,
            normalize=False,
            applyPCA=False,
            pretrain_num=args.pretrain_num,
            seed=seed
        )

        # Initialize model
        model = get_model(args)
        model.mode = "pretrain"
        model_name = model.__class__.__name__
        print('model name:', model_name)
        print('model mode:', model.mode)
        model.to(device)

        # Optimizer / scheduler / AMP scaler
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epoch - args.warmup_epochs
        )
        scaler = torch.cuda.amp.GradScaler()

        # Training state
        best_loss = float('inf')
        epochs_no_improve = 0
        saved_models = []

        for epoch in range(args.epoch):
            start_time = time.time()

            # Warmup LR (linear) then cosine schedule
            if epoch < args.warmup_epochs:
                lr_scale = min(1.0, float(epoch + 1) / args.warmup_epochs)
                for pg in optimizer.param_groups:
                    pg['lr'] = args.lr * lr_scale
            else:
                scheduler.step()

            # One epoch of pretraining
            avg_loss = Pretrain(pretrain_loader, model, optimizer, scaler, epoch, device)

            # Epoch timing (for logging)
            epoch_time = time.time() - start_time
            # print(f"Epoch {epoch} finished in {epoch_time:.2f}s, loss {avg_loss:.6f}")

            # Track and save the best checkpoints
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0

                ckpt_path = save_model(model, optimizer, epoch, avg_loss, args)
                saved_models.append((avg_loss, epoch, ckpt_path))
                saved_models.sort(key=lambda x: x[0])  # keep lowest loss first

                # Keep top-k checkpoints only
                while len(saved_models) > args.keep_weights:
                    _, _, old_path = saved_models.pop()
                    if os.path.exists(old_path):
                        os.remove(old_path)
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve}/{args.patience} epochs")

            # Early stopping
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                print(f"Best loss: {best_loss:.6f} at epoch {saved_models[0][1]}")
                break

        # Final logging
        if epoch == args.epoch - 1:
            print(f"Completed all {args.epoch} epochs")
        print(f"Best models saved: {[os.path.basename(p[2]) for p in saved_models]}")