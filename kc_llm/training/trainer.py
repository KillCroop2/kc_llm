import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from kc_llm.data.data_loader import collate_fn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
import os


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_model(model, dataset, epochs, batch_size, learning_rate, device, rank, world_size, checkpoint_path,
                use_amp=True, gradient_accumulation_steps=4):
    if world_size > 1:
        setup(rank, world_size)

    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    model.train()

    print(f"Training on device: {device}, Rank: {rank}/{world_size}")
    print(f"Model is on CUDA: {next(model.parameters()).is_cuda}")
    print(f"Using Automatic Mixed Precision: {use_amp}")
    print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=4,
                                pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4,
                                pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    num_training_steps = len(dataloader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

    scaler = GradScaler(enabled=use_amp)

    best_eval_loss = float('inf')

    for epoch in range(epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        total_loss = 0
        model.train()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", disable=(rank != 0))

        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with autocast(enabled=use_amp):
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * gradient_accumulation_steps

            if rank == 0:
                progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

            # Evaluate after each epoch
            eval_loss, perplexity = evaluate_model(model, dataset, batch_size, device, rank, world_size)
            print(f"Evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.4f}")

            # Save the best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(model, optimizer, scheduler, epoch, eval_loss, checkpoint_path)
                print(f"New best model saved with eval loss: {eval_loss:.4f}")

    if world_size > 1:
        cleanup()

    return model


def evaluate_model(model, dataset, batch_size, device, rank, world_size):
    model.eval()
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=4,
                                pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4,
                                pin_memory=True)

    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=(rank != 0)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with autocast(enabled=True):
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

    if world_size > 1:
        # Gather losses from all processes
        losses = [torch.zeros_like(torch.tensor(total_loss)).to(device) for _ in range(world_size)]
        dist.all_gather(losses, torch.tensor(total_loss).to(device))
        total_loss = sum(losses).item()

    avg_loss = total_loss / (len(dataloader) * world_size)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}. Starting from scratch.")
        return 0, float('inf')

    checkpoint = torch.load(path, map_location=device)
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    return checkpoint['epoch'], checkpoint['loss']