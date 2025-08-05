import torch
import tqdm

lr = 5e-4
load_model = False
save_model = False

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

def train_stg1(model, train_dl, optimizer, accelerator, epochs, load_model=False, save_model=True, checkpoint_path=None):
    if load_model:
        load_checkpoint(model, optimizer, checkpoint_path)
        model.float()  # Convert model back to float precision if needed
    # Training loop
    model.train()
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch+1}/{epochs}")
        total_loss = 0
        loop = tqdm(train_dl, leave=True)

        for idx, batch in enumerate(loop):
            # Move batch to device
            pixel_values = batch["pixel_values"]
            dino_pixel_values = batch["dino_pixel_values"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            MeSH_input_ids = batch["MeSH_input_ids"]
            MeSH_attn_mask = batch["MeSH_attention_mask"]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with accelerator.autocast():
                MeSH_outputs = model(
                    pixel_values=pixel_values,
                    dino_pixel_values=dino_pixel_values,
                    input_ids=MeSH_input_ids, 
                    attention_mask=MeSH_attn_mask, 
                    labels=MeSH_input_ids
                )
                loss = MeSH_outputs.loss

            # Update progress bar
            loop.set_description(f"Loss: {loss.item():.4f}")
            total_loss += loss.item()

            # Backward pass with gradient scaling
            accelerator.backward(loss)
            optimizer.step()

        avg_loss = total_loss / len(train_dl)
        print(f"Loss: {avg_loss:.4f}")
        
        if save_model and (epoch+1)%5==0:
            save_checkpoint(
                model=model.half(),  # Convert to half precision for storage
                optimizer=optimizer,
                epoch=epoch+1,
                loss=avg_loss
            )

def train_stg2(model, train_dl, optimizer, accelerator, epochs, alpha=0.8, load_model=False, save_model=True, checkpoint_path=None):
    # Load checkpoint if specified
    if load_model:
        load_checkpoint(model, optimizer, checkpoint_path)
        model.float()  # Convert model back to float precision if needed

    # Training loop
    model.train()
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        total_loss = 0
        loop = tqdm(train_dl, leave=True)
        
        for idx, batch in enumerate(loop):
            # Move batch to device
            pixel_values = batch["pixel_values"]
            dino_pixel_values = batch["dino_pixel_values"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            MeSH_input_ids = batch["MeSH_input_ids"]
            MeSH_attn_mask = batch["MeSH_attention_mask"]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with accelerator.autocast():
                outputs = model(
                    pixel_values=pixel_values,
                    dino_pixel_values=dino_pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                MeSH_outputs = model(
                    pixel_values=pixel_values,
                    dino_pixel_values=dino_pixel_values,
                    input_ids=MeSH_input_ids, 
                    attention_mask=MeSH_attn_mask, 
                    labels=MeSH_input_ids
                )
                loss = alpha * outputs.loss + (1-alpha) * MeSH_outputs.loss
            
            # Update progress bar
            loop.set_description(f"Loss: {loss.item():.4f}")
            total_loss += loss.item()
            
            # Backward pass with gradient scaling
            accelerator.backward(loss)
            optimizer.step()
        
        # Average loss for the epoch
        avg_loss = total_loss / len(train_dl)
        print(f"Loss: {avg_loss:.4f}")

        # Save checkpoint
        if save_model and (epoch+1)%5==0:
            save_checkpoint(
                model=model.half(),  # Convert to half precision for storage
                optimizer=optimizer,
                epoch=10 + epoch + 1,
                loss=avg_loss
            )