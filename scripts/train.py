import argparse
import torch
import torch.multiprocessing as mp
import os
from kc_llm import GPTModel, load_data, train_model, load_tokenizer, get_vocab_size, load_checkpoint
from pathlib import Path

import os
import torch.distributed as dist


def setup_distributed_env(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def run_training(rank, world_size, args):
    setup_distributed_env(rank, world_size)

    try:
        device = torch.device(f"cuda:{rank}")
        print(f"Rank {rank}: Using device {device}")

        tokenizer = load_tokenizer()
        print(f"Rank {rank}: Tokenizer loaded")

        dataset = load_data(args.data_file, tokenizer, args.max_length)
        print(f"Rank {rank}: Dataset loaded, size: {len(dataset)}")

        vocab_size = get_vocab_size(tokenizer)
        model = GPTModel(vocab_size)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        print(f"Rank {rank}: Model created and wrapped with DDP")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Rank {rank}: Checkpoint directory created")

        start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, args.checkpoint_path, device)
        print(f"Rank {rank}: Checkpoint loaded, starting from epoch {start_epoch}")

        print(f"Rank {rank}: Starting training")
        trained_model = train_model(
            model,
            dataset,
            args.epochs - start_epoch,
            args.batch_size,
            args.learning_rate,
            device,
            rank,
            world_size,
            args.checkpoint_path,
            use_amp=args.use_amp,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        print(f"Rank {rank}: Training completed")

        if rank == 0:
            print(f"Best model saved as {args.checkpoint_path}")

    except Exception as e:
        print(f"Rank {rank}: An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train the WikiGPT model")
    parser.add_argument("--data_file", type=str, default="training_data.json", help="Path to the data file")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--checkpoint_path", type=str, default="best_model.pth",
                        help="Path to save/load the best model checkpoint")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Using {world_size} GPUs!")
        mp.spawn(run_training, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("No multiple GPUs found. Running on a single GPU or CPU.")
        run_training(0, 1, args)


if __name__ == "__main__":
    main()
