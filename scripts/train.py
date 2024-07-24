import argparse
import torch
import torch.multiprocessing as mp
import os
from kc_llm import GPTModel, train_model, load_tokenizer, get_vocab_size, \
    load_checkpoint, setup, cleanup
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from kc_llm.data import load_data, EfficientDataLoader
from torch.utils.data import DistributedSampler

def setup_distributed_env():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

def run_training(rank, world_size, args):
    if world_size > 1:
        setup_distributed_env()
        setup(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer()
    train_dataset = load_data(args.data_file, tokenizer, args.max_length)

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_dataloader = EfficientDataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                               sampler=train_sampler, num_workers=4, pin_memory=True)
    else:
        train_dataloader = EfficientDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True)

    vocab_size = get_vocab_size(tokenizer)
    model = GPTModel(vocab_size)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    total_steps = len(train_dataset) * args.epochs // (args.batch_size * world_size * args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load checkpoint if it exists
    start_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.checkpoint_path, device)

    train_model(
        model,
        train_dataloader,
        args.epochs - start_epoch,  # Adjust epochs if resuming
        device,
        rank,
        world_size,
        args.checkpoint_path,
        optimizer,
        scheduler,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    if rank == 0:
        print(f"Training completed. Final model saved as {args.checkpoint_path}")

    if world_size > 1:
        cleanup()

def main():
    parser = argparse.ArgumentParser(description="Train the WikiGPT model")
    parser.add_argument("--data_file", type=str, default="improved_training_data.json", help="Path to the data file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--checkpoint_path", type=str, default="best_model.pth",
                        help="Path to save/load the model checkpoint")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    args = parser.parse_args()

    # Set up multi-GPU training
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Using {world_size} GPUs!")
        mp.spawn(run_training, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("Running on a single GPU or CPU.")
        run_training(0, 1, args)

if __name__ == "__main__":
    main()