import logging
import os
import pickle
import shutil
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from utils.config import load_config
from utils.data_handling import (cache_processed_data, load_cached_data,
                                 open_datafiles, prepare_dataloader,
                                 scale_muon_data)
from utils.flow_models import NormalizingFlow
from utils.logging_config import setup_logging
from utils.plotting import plot_feature_histograms
from utils.run_management import get_next_attempt_number
from utils.training import train_model

logger = logging.getLogger(__name__)

def main():
    # Load external configuration
    config = load_config()

    # Setup run folder
    run_base_dir = config.get("run_base_dir", "outputs")
    attempt_number, run_dir = get_next_attempt_number(run_base_dir)

    config_copy_path = os.path.join(run_dir, "config.yaml")
    shutil.copy("config.yaml", config_copy_path)

    # Logging setup
    setup_logging(config.get("logging", {}), run_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"Run {attempt_number} logging to {run_dir}")

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    datapath_mothers = config.get("datapath_mothers")
    model_config = config.get("model", {})
    model_dir = run_dir
    name_model = model_config.get("name", "RealNVP_Flow")

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))

    # Check for cached data
    cache_path = os.path.join(run_dir, "processed_train_data.pt")
    cached_data = load_cached_data(cache_path)

    if cached_data is None:
        mothers_df = open_datafiles(datapath_mothers)
        print('Loading a smaller sample size.')
        scaled_np, scaler_mother, raw_np = scale_muon_data(mothers_df, run_dir)

        indices = np.arange(raw_np.shape[0])
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        train_scaled = scaled_np[train_idx]

        cache_processed_data(cache_path, train_scaled)
    else:
        train_scaled = cached_data.numpy()

    # Split data - training and validation
    indices = np.arange(raw_np.shape[0])
    split_config = config.get("data", {})
    val_ratio = split_config.get("val_ratio", 0.2)
    
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=42)
    train_scaled = scaled_np[train_idx]
    val_scaled = scaled_np[val_idx]

    batch_size = config.get("batch_size", 128)
    train_loader = DataLoader(TensorDataset(torch.tensor(train_scaled, dtype=torch.float)),
                              batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_scaled, dtype=torch.float)),
                            batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # Initialise the NF model
    input_dim = train_scaled.shape[1]
    hidden_dim = model_config.get("hidden_dim", 64)
    n_layers = model_config.get("n_layers", 8)
    flow_model = NormalizingFlow(input_dim, hidden_dim, n_layers)

    # total_params = sum(p.numel() for p in flow_model.parameters())
    # trainable_params = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    # print(f"Total parameters: {total_params}")
    # print(f"Trainable parameters: {trainable_params}")

    # Train the model
    epochs = model_config.get("epochs", 150)
    lr = model_config.get("learning_rate", 0.001)
    logger.info("-- Starting training --")
    trained_model = train_model(
        model=flow_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        writer=writer,
        device=device,
        model_dir=model_dir,
        name_model=name_model,
        patience=model_config.get("patience", 20),
        epochs_no_improve=0
    )

    # Evaluate the model by generating synthetic data
    trained_model.eval()
    with torch.no_grad():

        num_samples = val_scaled.shape[0]
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Generate latent space sample
        z = trained_model.base_dist.sample((num_samples,)).to(device)
        # Generate synthetic data
        generated_scaled = trained_model.inverse(z)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

    generated_scaled = generated_scaled.cpu().numpy()

    total_time = end_time - start_time
    time_per_event = total_time / num_samples
    logger.info(f"Total generation time: {total_time:.6f} s.")
    logger.info(f"Time per event: {time_per_event:.6f} s.")

    # Inverse-transform data to original space
    generated_original = scaler_mother.inverse_transform(generated_scaled)
    val_original = scaler_mother.inverse_transform(val_scaled)

    # Save generated features
    gen_file = os.path.join(run_dir, "generated_muon_features.pkl")
    with open(gen_file, "wb") as f:
        pickle.dump({"generated_scaled": generated_scaled,
                     "generated_original": generated_original}, f)
    logger.info(f"Saved generated features to {gen_file}")

    # Plot histograms comparing true and generated data
    plot_feature_histograms(val_scaled, generated_scaled, run_dir, space_tag="scaled")
    plot_feature_histograms(val_original, generated_original, run_dir, space_tag="original")

    writer.close()
    logger.info("-- Training and evaluation complete --[]")

if __name__ == "__main__":
    main()
