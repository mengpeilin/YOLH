#!/usr/bin/env python3
"""Train YOLH on a prebuilt dataset."""

import argparse
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import MinkowskiEngine as ME
from tqdm import tqdm

from dataset.yolh_dataset import YolhDataset, collate_fn
PROJECT_ROOT = Path(__file__).resolve().parent
from policy.yolh import YOLH
from policy.utils.training import set_seed, plot_history, sync_loss

try:
    from diffusers.optimization import get_cosine_schedule_with_warmup
except ImportError:
    get_cosine_schedule_with_warmup = None

def train(args):
    torch.multiprocessing.set_sharing_strategy("file_system")
    use_ddp = "WORLD_SIZE" in os.environ
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

    if use_ddp and WORLD_SIZE > 1:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=WORLD_SIZE, rank=RANK,
        )
    elif use_ddp:
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=WORLD_SIZE, rank=RANK,
        )

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if RANK == 0:
        print("Loading dataset ...")
    dataset = YolhDataset(
        dataset_path=args.dataset,
        voxel_size=args.voxel_size,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty - check --dataset")

    if WORLD_SIZE > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True,
        )
    else:
        sampler = None

    per_gpu_bs = max(1, args.batch_size // WORLD_SIZE)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_gpu_bs,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
    )

    if RANK == 0:
        print("Building YOLH policy ...")
    policy = YOLH(
        num_action=args.num_action,
        input_dim=6,                            # xyz + rgb_norm
        obs_feature_dim=args.obs_feature_dim,
        action_dim=10,                          # pos(3)+rot6d(6)+width(1)
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
    ).to(device)

    if RANK == 0:
        n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"Parameters: {n_params / 1e6:.2f}M")

    if WORLD_SIZE > 1:
        policy = nn.parallel.DistributedDataParallel(
            policy, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,
            find_unused_parameters=True,
        )

    if args.resume_ckpt is not None:
        state = torch.load(args.resume_ckpt, map_location=device)
        mod = policy.module if WORLD_SIZE > 1 else policy
        mod.load_state_dict(state, strict=False)
        if RANK == 0:
            print(f"Resumed from {args.resume_ckpt}")

    if RANK == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr,
        betas=(0.95, 0.999), weight_decay=1e-6,
    )

    total_steps = len(dataloader) * args.num_epochs
    if get_cosine_schedule_with_warmup is not None:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=min(2000, total_steps // 5),
            num_training_steps=total_steps,
        )
        lr_scheduler.last_epoch = len(dataloader) * (args.resume_epoch + 1) - 1
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps,
        )
        if RANK == 0:
            print("(diffusers not installed - using CosineAnnealingLR)")

    train_history = []
    policy.train()

    for epoch in range(args.resume_epoch + 1, args.num_epochs):
        if RANK == 0:
            print(f"Epoch {epoch}")
        if sampler is not None:
            sampler.set_epoch(epoch)

        optimizer.zero_grad()
        num_steps = len(dataloader)
        pbar = tqdm(dataloader, desc=f"epoch {epoch}") if RANK == 0 else dataloader
        avg_loss = 0.0

        for data in pbar:
            cloud_coords = data["input_coords_list"]
            cloud_feats = data["input_feats_list"]
            action_data = data["action_normalized"]

            cloud_feats = cloud_feats.to(device)
            cloud_coords = cloud_coords.to(device)
            action_data = action_data.to(device)
            cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)

            loss = policy(cloud_data, action_data, batch_size=action_data.shape[0])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            avg_loss += loss.item()

        avg_loss /= max(num_steps, 1)
        if WORLD_SIZE > 1:
            sync_loss(avg_loss, device)
        train_history.append(avg_loss)

        if RANK == 0:
            print(f"  Train loss: {avg_loss:.6f}")
            if (epoch + 1) % args.save_epochs == 0:
                mod = policy.module if WORLD_SIZE > 1 else policy
                torch.save(
                    mod.state_dict(),
                    os.path.join(
                        args.ckpt_dir,
                        f"policy_epoch_{epoch + 1}_seed_{args.seed}.ckpt",
                    ),
                )
                plot_history(train_history, epoch, args.ckpt_dir, args.seed)

    if RANK == 0:
        mod = policy.module if WORLD_SIZE > 1 else policy
        torch.save(mod.state_dict(), os.path.join(args.ckpt_dir, "policy_last.ckpt"))
        print("Training complete.")

def main():
    parser = argparse.ArgumentParser(description="Train YOLH on pipeline data")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to train_dataset.npz generated by 06_generate_dataset.py")
    parser.add_argument("--num-action", type=int, default=20,
                        help="Number of future action steps to predict")
    parser.add_argument("--voxel-size", type=float, default=0.005,
                        help="ME sparse voxel size in metres")

    parser.add_argument("--obs-feature-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--num-decoder-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Directory to save checkpoints")
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--resume-epoch", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--save-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=233)

    train(parser.parse_args())


if __name__ == "__main__":
    main()
