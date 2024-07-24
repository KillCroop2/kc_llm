import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_model(model, dataloader, epochs, device, rank, world_size, checkpoint_path,
                optimizer, scheduler, use_amp=True, gradient_accumulation_steps=1):
    model.train()
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        
        total_loss = 0
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
            save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, checkpoint_path)

    return model

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