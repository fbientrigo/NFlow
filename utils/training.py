# training.py

"""
Training utilities for RealNVP flows with a learnable Gaussian Mixture prior.

This module provides:
- GuidedFlowLoss: a semi-supervised loss combining global NLL with class-specific components.
- FlowNLL: a standard negative log-likelihood loss wrapper.
- collect_latents: extract latent representations from a trained flow.
- plot_latent_dims: visualize marginal histograms of each latent dimension.
- latent_metrics: compute silhouette score and Mahalanobis distance between components.
- train_model: a flexible training loop supporting NLL or guided loss, LR scheduling, early stopping, pruning, and inline diagnostics.
"""

import math
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import optuna
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class GuidedFlowLoss(nn.Module):
    """
    Combined loss for guided training of a flow with a GMM prior.

    This loss consists of:
      1) Global negative log-likelihood under the GMM prior.
      2) A sub-loss for "lvl1" samples pushing them toward component 0 (mean mu[0], std σ1).
      3) A sub-loss for "lvl2" samples pushing them toward component 1 (mean mu[1], std σ2).

    Parameters
    ----------
    flow : nn.Module
        Invertible flow model with methods `forward(x)` returning (z, logdet).
    prior : nn.Module
        MixturePrior defining `log_prob(z)` and `mu` parameters.
    λ1 : float
        Weight for the lvl1 sub-loss.
    σ1 : float
        Standard deviation used when computing log-prob for component 0.
    λ2 : float
        Weight for the lvl2 sub-loss.
    σ2 : float
        Standard deviation used when computing log-prob for component 1.
    """
    def __init__(self, flow, prior, λ1, σ1, λ2, σ2):
        super().__init__()
        self.flow = flow
        self.prior = prior
        self.λ1 = λ1
        self.σ1 = σ1
        self.λ2 = λ2
        self.σ2 = σ2

    def forward(self, x, mask_lvl2):
        """
        Compute the guided flow loss.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, input_dim)
            Input data batch.
        mask_lvl2 : torch.BoolTensor of shape (batch_size,)
            Boolean mask indicating which samples belong to lvl2.

        Returns
        -------
        loss : torch.Tensor, scalar
            The combined loss value.
        """
        z, logdet = self.flow.forward(x)

        # 1) Global NLL term
        nll_global = - (self.prior.log_prob(z) + logdet).mean()

        # 2) Log-probabilities under each component Gaussian
        dist0 = torch.distributions.Normal(self.prior.mu[0], self.σ1)
        dist1 = torch.distributions.Normal(self.prior.mu[1], self.σ2)
        lp0 = dist0.log_prob(z).sum(dim=1)
        lp1 = dist1.log_prob(z).sum(dim=1)

        # 3) Sub-losses for each class
        loss0 = -(lp0[~mask_lvl2].mean()) if (~mask_lvl2).any() else torch.tensor(0., device=z.device)
        loss1 = -(lp1[ mask_lvl2].mean()) if ( mask_lvl2).any() else torch.tensor(0., device=z.device)

        return nll_global + self.λ1 * loss0 + self.λ2 * loss1


class FlowNLL(nn.Module):
    """
    Negative log-likelihood loss wrapper for any prior with a `log_prob(z)` method.

    Parameters
    ----------
    flow : nn.Module
        Invertible flow model with methods `forward(x)` returning (z, logdet).
    prior : nn.Module
        Prior distribution with method `log_prob(z)`.
    """
    def __init__(self, flow, prior):
        super().__init__()
        self.flow = flow
        self.prior = prior

    def forward(self, x):
        """
        Compute the NLL loss.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, input_dim)
            Input data batch.

        Returns
        -------
        loss : torch.Tensor, scalar
            The mean negative log-likelihood over the batch.
        """
        z, logdet = self.flow.forward(x)
        return - (self.prior.log_prob(z) + logdet).mean()


def collect_latents(model, loader, device):
    """
    Collect latent representations and masks from a data loader.

    Parameters
    ----------
    model : nn.Module
        Trained flow model with `forward(x)` → (z, logdet).
    loader : DataLoader
        Yields (x, mask_lvl2) tuples.
    device : torch.device
        Device on which to perform inference.

    Returns
    -------
    zs : torch.Tensor of shape (n_samples, latent_dim)
        Concatenated latent codes for all samples.
    ys : torch.Tensor of shape (n_samples,)
        Concatenated boolean masks for lvl2.
    """
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z, _ = model.forward(x)
            zs.append(z.cpu())
            ys.append(y)
    return torch.cat(zs), torch.cat(ys)


def plot_latent_dims(zs, ys, prior):
    """
    Plot marginal histograms of each latent dimension, with component means.

    Parameters
    ----------
    zs : array-like of shape (n_samples, latent_dim)
        Latent codes, either ndarray or torch.Tensor.
    ys : array-like of shape (n_samples,)
        Boolean mask for lvl2 membership.
    prior : MixturePrior
        Prior containing `mu` (shape (K, latent_dim)) and `K`.
    """
    if isinstance(zs, torch.Tensor):
        zs = zs.cpu().detach().numpy()
    if isinstance(ys, torch.Tensor):
        ys = ys.cpu().detach().numpy()
    ys = ys.astype(bool)

    N, D = zs.shape
    fig, axes = plt.subplots(1, D, figsize=(4 * D, 4))
    if D == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.hist(zs[~ys, i], bins=30, alpha=0.5, density=True, label='lvl1', color='C0')
        ax.hist(zs[ ys, i], bins=30, alpha=0.5, density=True, label='lvl2', color='C1')
        for k in range(prior.K):
            mu_ki = prior.mu[k, i].item()
            ax.axvline(mu_ki, linestyle='--', color=f'C{k}', label=f'μ_{k}, dim{i}')
        ax.set_xlabel(f'z[{i}]')
        ax.set_ylabel('Density')
        ax.set_title(f'Latent Dimension #{i}')
        ax.legend(fontsize='small')
    plt.tight_layout()
    plt.show()


def latent_metrics(zs, ys, prior):
    """
    Compute clustering quality metrics in latent space.

    Parameters
    ----------
    zs : torch.Tensor or ndarray of shape (n_samples, latent_dim)
        Latent codes.
    ys : torch.Tensor or ndarray of shape (n_samples,)
        Boolean mask for lvl2 membership.
    prior : MixturePrior
        Prior with `mu` and `log_sig` parameters, and attribute `K`.

    Returns
    -------
    sil : float
        Silhouette score of the latent embeddings using `ys` as labels.
    maha : float or None
        Mahalanobis distance between the two component means if `K == 2`,
        otherwise None.
    """
    labels = ys.numpy() if isinstance(ys, torch.Tensor) else ys
    data   = zs.numpy() if isinstance(zs, torch.Tensor) else zs
    sil = silhouette_score(data, labels)

    maha = None
    if prior.K == 2:
        diff = (prior.mu[0] - prior.mu[1]).unsqueeze(0)
        cov = (prior.log_sig.exp()[0]**2 + prior.log_sig.exp()[1]**2).mean()
        maha = (diff.norm() / math.sqrt(cov)).item()

    return sil, maha


def train_model(model,
                train_loader,
                val_loader,
                prior,
                epochs: int,
                lr: float,
                writer,
                device,
                model_dir: str,
                name_model: str,
                alpha: float = 0.5,
                patience: int = 20,
                weight_decay: float = 0.0,
                trial=None,
                mod_epochs: int = 10,
                kind: str = 'nll',
                hyperparams: dict = None):
    """
    Train a RealNVP flow model with a Gaussian Mixture prior.

    This function supports both:
      - Standard NLL training.
      - Guided training combining global NLL and class-specific losses.

    Parameters
    ----------
    model : nn.Module
        The RealNVP flow model.
    train_loader : DataLoader
        Yields (x, mask_lvl2) for training.
    val_loader : DataLoader
        Yields (x, mask_lvl2) for validation.
    prior : MixturePrior
        Learnable mixture prior with `mu`, `log_sig`, `log_pi`.
    epochs : int
        Number of training epochs.
    lr : float
        Base learning rate for flow parameters.
    writer : SummaryWriter
        TensorBoard writer for logging.
    device : torch.device
        Device for computation.
    model_dir : str
        Directory to save model checkpoints.
    name_model : str
        Base filename for saved model.
    alpha : float, default=0.5
        Sampling proportion for component 1 in diagnostic plots.
    patience : int, default=20
        Early stopping patience (epochs without improvement).
    weight_decay : float, default=0.0
        L2 regularization coefficient.
    trial : optuna.Trial, optional
        Optuna trial for pruning (default: None).
    mod_epochs : int, default=10
        Interval (in epochs) for inline diagnostic plots.
    kind : {'nll', 'guided'}, default='nll'
        Loss mode: `'nll'` for FlowNLL, `'guided'` for GuidedFlowLoss.
    hyperparams : dict, optional
        Hyperparameters for guided loss (`lambda1`, `sigma1`, `lambda2`, `sigma2`).

    Returns
    -------
    nn.Module
        The model loaded with the best validation performance.

    Notes
    -----
    - Uses `CosineAnnealingLR` to decay learning rate from `lr` to `lr * 1e-3`.
    - Supports per-parameter learning rates: `prior.mu` uses `lr * 5`,
      `prior.log_sig` and `prior.log_pi` use `lr * 0.5`.
    - Early stopping monitors validation loss only.
    - Integrates Optuna pruning if `trial` is provided.
    """
    model.to(device)
    prior.to(device)

    # Select loss function
    if kind == 'nll':
        loss_fn = FlowNLL(model, prior)
    elif kind == 'guided':
        hp = hyperparams or {}
        loss_fn = GuidedFlowLoss(
            model, prior,
            hp['lambda1'], hp['sigma1'],
            hp['lambda2'], hp['sigma2']
        )
    else:
        raise ValueError(f"Unknown kind: {kind}")

    # Optimizer with differential learning rates
    optimizer = optim.Adam([
        {'params': model.parameters(),               'lr': lr},
        {'params': [prior.mu],                       'lr': lr * 5},
        {'params': [prior.log_sig, prior.log_pi],    'lr': lr * 0.5},
    ], weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 1e-3
    )

    best_val = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # —— Training phase ——
        model.train()
        prior.train()
        for x, mask in train_loader:
            x, mask = x.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = loss_fn(x) if kind == 'nll' else loss_fn(x, mask)
            loss.backward()
            optimizer.step()

        # —— Evaluate train loss in eval mode ——
        model.eval()
        prior.eval()
        train_eval = 0.0
        with torch.no_grad():
            for x, mask in train_loader:
                x, mask = x.to(device), mask.to(device)
                train_eval += (loss_fn(x) if kind == 'nll' else loss_fn(x, mask)).item()
        train_losses.append(train_eval / len(train_loader))

        # —— Validation phase ——
        running_val = 0.0
        with torch.no_grad():
            for x, mask in val_loader:
                x, mask = x.to(device), mask.to(device)
                running_val += (loss_fn(x) if kind == 'nll' else loss_fn(x, mask)).item()
        val_losses.append(running_val / len(val_loader))

        # Log metrics and learning rate
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalars("Loss", {"Train": train_losses[-1], "Val": val_losses[-1]}, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        logger.info(f"[{epoch}/{epochs}] LR={current_lr:.2e}  "
                    f"Train={train_losses[-1]:.4f}  Val={val_losses[-1]:.4f}")

        # —— Inline diagnostics every mod_epochs —— 
        if mod_epochs and epoch % mod_epochs == 0:
            clear_output(wait=True)
            # Plot loss curves
            plt.plot(range(1, epoch + 1), train_losses, label="Train")
            plt.plot(range(1, epoch + 1), val_losses,   label="Val")
            plt.legend()
            plt.grid(True)
            plt.show()
            # Additional diagnostic plots omitted for brevity

        # —— Early stopping & checkpointing ——
        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{model_dir}/{name_model}.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Optuna pruning
        if trial:
            trial.report(val_losses[-1], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Step the scheduler
        scheduler.step()

    # Load best model
    model.load_state_dict(torch.load(f"{model_dir}/{name_model}.pt", map_location=device))
    logger.info(f"Training complete. Best val loss: {best_val:.4f}")
    return model

