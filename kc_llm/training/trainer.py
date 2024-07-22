import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from kc_llm.data.data_loader import collate_fn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup


def train_model(model, dataset, epochs, batch_size, learning_rate, device, use_amp=True, gradient_accumulation_steps=16):
    model.to(device)
    model.train()

    print(f"Training on device: {device}")
    print(f"Model is on CUDA: {next(model.parameters()).is_cuda}")
    print(f"Using Automatic Mixed Precision: {use_amp}")
    print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
                            pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    num_training_steps = len(dataloader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

    scaler = GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
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

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # Evaluate after each epoch
        eval_loss, perplexity = evaluate_model(model, dataset, batch_size, device)
        print(f"Evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.4f}")

    return model


def evaluate_model(model, dataset, batch_size, device):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
                            pin_memory=True)
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with autocast(enabled=True):
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity