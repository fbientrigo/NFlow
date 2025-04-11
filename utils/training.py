import logging
import os

import optuna
import torch
import torch.optim as optim

logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, epochs, lr, writer, device, model_dir, name_model, patience=20, epochs_no_improve=0, weight_decay=0.0, trial=None):

    best_val_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    
    # Reinitialize the base distribution on the target device
    input_dim = model.layers[0].input_dim
    model.base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(input_dim, device=device),
        torch.eye(input_dim, device=device)
    )
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            loss = -model.log_prob(x).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                loss = -model.log_prob(x).mean()
                val_loss += loss.item()
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")
        
        writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)
        
        # For Optuna
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                logger.info("Pruning trial at epoch %d", epoch)
                raise optuna.TrialPruned()

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_path = os.path.join(model_dir, f"{name_model}.pt")
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            logger.info("Early stopping triggered!")
            break

    logger.info(f"Training complete. Model saved as {os.path.join(model_dir, f'{name_model}.pt')}.")
    # Load best checkpoint
    model.load_state_dict(torch.load(os.path.join(model_dir, f"{name_model}.pt"), map_location=device))

    return model
