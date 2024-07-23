import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from kc_llm import GPTModel, load_data, load_tokenizer, get_vocab_size
import os
from pathlib import Path


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def save_checkpoint(model, optimizer, epoch, loss, args, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    if is_best:
        best_model_path = Path(args.checkpoint_dir) / "best_model.pth"
        torch.save(checkpoint, best_model_path)
        print(f"Best model saved: {best_model_path}")


def load_checkpoint(model, optimizer, args):
    checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{args.resume_epoch}.pth"
    if not checkpoint_path.exists():
        raise ValueError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    print(f"Resuming from epoch {start_epoch}, loss: {loss}")
    return model, optimizer, start_epoch, loss


def train(rank, world_size, args):
    if world_size > 1:
        setup(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = load_tokenizer()
    dataset = load_data(args.data_file, tokenizer, args.max_length)

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    vocab_size = get_vocab_size(tokenizer)
    model = GPTModel(vocab_size).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    best_loss = float('inf')

    if args.resume_epoch is not None:
        model, optimizer, start_epoch, best_loss = load_checkpoint(model, optimizer, args)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if world_size > 1:
            sampler.set_epoch(epoch)
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if rank == 0 or world_size == 1:
            print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

            is_best = avg_loss < best_loss
            best_loss = min(avg_loss, best_loss)

            save_checkpoint(model, optimizer, epoch + 1, avg_loss, args, is_best)

    if rank == 0 or world_size == 1:
        final_model_path = Path(args.output_model)
        if world_size > 1:
            torch.save(model.module.state_dict(), final_model_path)
        else:
            torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved as {final_model_path}")

    if world_size > 1:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="Train the WikiGPT model")
    parser.add_argument("--data_file", type=str, default="training_data.json", help="Path to the data file")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--output_model", type=str, default="wikigpt_model.pth", help="Path to save the trained model")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_epoch", type=int, help="Epoch to resume training from")
    args = parser.parse_args()

    # Create checkpoint directory if it doesn't exist
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Using {world_size} GPUs!")
        mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("No multiple GPUs found. Running on a single GPU or CPU.")
        train(0, 1, args)


if __name__ == "__main__":
    main()