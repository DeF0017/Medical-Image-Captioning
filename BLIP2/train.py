import torch
import tqdm

lr = 5e-4
load_model = True
save_model = True

autocast = torch.amp.autocast('cuda', enabled=True, dtype=torch.half)
scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=4096)

def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, lr, filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f"Checkpoint loaded from {filename}, epoch {epoch}, loss {loss}")

def train(model, optimizer, train_dl, epoch):
    if load_model:
        load_checkpoint(model, optimizer, lr, "/kaggle/input/medcapcheckpoints/checkpoint-2.pth")
        model.float()
    model.train()
    for epoch in range(epoch):
        print("Epoch: ", epoch+1)
        total_loss = 0
        loop =tqdm(train_dl, leave=True)
        for idx, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(model.device)
            pixel_vals = batch["pixel_values"].to(model.device, torch.float16)
            optimizer.zero_grad()
            with autocast:
                outputs = model(input_ids=input_ids, pixel_values=pixel_vals, 
                                labels=input_ids)
                loss = outputs.loss
            loop.set_description(f"Loss: {loss.item():.4f}")
            total_loss += loss.item()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() 
            
        # Average the loss over all batches
        avg_loss = total_loss / len(train_dl)  # Dividing by number of batches
        print("Loss:", avg_loss)
        
        if save_model and (epoch+1)%10==0:
            save_checkpoint(model.half(), optimizer, epoch+1, avg_loss, f"checkpoint.pth")
