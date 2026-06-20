
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

logger = logging.getLogger(__name__)

DEFAULT_PREDICTION_HORIZONS = (15, 60, 240, 480)


class LSTMEncoder(nn.Module):
    """LSTM Encoder for VAE."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        latent_dim: int = 64,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Latent space parameters
        encoder_output_dim = hidden_dim * self.num_directions
        self.mu_layer = nn.Linear(encoder_output_dim, latent_dim)
        self.logvar_layer = nn.Linear(encoder_output_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            hidden: Last hidden state
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state (concatenate both directions if bidirectional)
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]
        
        hidden = self.dropout(hidden)
        
        # Compute latent parameters
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        
        return mu, logvar, lstm_out


class LSTMDecoder(nn.Module):
    """LSTM Decoder for VAE."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        seq_len: int = 100,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        # Project latent vector to initial hidden state
        self.hidden_proj = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.cell_proj = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            seq_len: Sequence length (uses default if None)
        
        Returns:
            reconstructed: Reconstructed sequence
        """
        seq_len = seq_len or self.seq_len
        batch_size = z.size(0)
        
        # Initialize hidden states from latent vector
        hidden = self.hidden_proj(z).view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.transpose(0, 1).contiguous()
        
        cell = self.cell_proj(z).view(batch_size, self.num_layers, self.hidden_dim)
        cell = cell.transpose(0, 1).contiguous()
        
        # Create sequence of latent vectors
        z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)
        
        # LSTM forward
        lstm_out, _ = self.lstm(z_seq, (hidden, cell))
        
        # Output layer
        reconstructed = self.output_layer(lstm_out)
        
        return reconstructed


class LSTMVAE(pl.LightningModule):
    """
    LSTM Variational Autoencoder for temporal anomaly detection.
    
    Combines LSTM encoder-decoder architecture with VAE framework
    to learn a probabilistic latent representation of time series data.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        latent_dim: int = 64,
        dropout: float = 0.2,
        bidirectional: bool = True,
        beta: float = 0.001,
        kl_annealing: bool = True,
        kl_annealing_steps: int = 10000,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        seq_len: int = 100,
        prediction_horizons: Tuple[int, ...] = DEFAULT_PREDICTION_HORIZONS,
        pos_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.beta = beta
        self.kl_annealing = kl_annealing
        self.kl_annealing_steps = kl_annealing_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seq_len = seq_len
        self.prediction_horizons = tuple(prediction_horizons)
        self.pos_weights = pos_weights  # Class imbalance: {horizon: positive_weight}
        
        # Encoder and Decoder
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            latent_dim=latent_dim,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            seq_len=seq_len,
            dropout=dropout
        )
        
        # Event prediction head
        self.event_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3 classes: normal, tsunami, volcano
        )

        self.horizon_event_heads = nn.ModuleDict({
            str(horizon): nn.Linear(latent_dim, 2)
            for horizon in self.prediction_horizons
        })
        
        # Class imbalance handling: configurable positive weights for horizon binary heads
        # Weights are stored in pos_weights dict passed to __init__
        
        # Anomaly score head
        self.anomaly_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Training metrics
        self.train_reconstruction_loss = []
        self.train_kl_loss = []
        self.val_reconstruction_loss = []
        self.val_kl_loss = []
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between posterior and prior."""
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl
    
    def compute_reconstruction_loss(
        self,
        x: torch.Tensor,
        x_reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss (MSE)."""
        return F.mse_loss(x_reconstructed, x, reduction='mean')
    
    def get_beta(self, step: int) -> float:
        """Get beta value with optional KL annealing."""
        if not self.kl_annealing:
            return self.beta
        
        # Linear annealing
        return min(self.beta, self.beta * (step / self.kl_annealing_steps))
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Dictionary containing:
                - x_reconstructed: Reconstructed input
                - mu: Mean of latent distribution
                - logvar: Log variance of latent distribution
                - z: Sampled latent vector
                - event_probs: Event probabilities
                - anomaly_score: Anomaly score
        """
        # Encode
        mu, logvar, lstm_out = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_reconstructed = self.decoder(z)
        
        # Event prediction. The multi-class head is kept for compatibility, while
        # horizon heads provide P(tsunami|X) and P(volcano|X) per target horizon.
        event_logits = self.event_head(z)
        event_probs = F.softmax(event_logits, dim=-1)
        horizon_event_logits = {
            horizon: head(z)
            for horizon, head in self.horizon_event_heads.items()
        }
        horizon_event_probs = {
            horizon: torch.sigmoid(logits)
            for horizon, logits in horizon_event_logits.items()
        }
        
        # Anomaly score
        anomaly_score = self.anomaly_head(z)
        
        return {
            'x_reconstructed': x_reconstructed,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'event_logits': event_logits,
            'event_probs': event_probs,
            'horizon_event_logits': horizon_event_logits,
            'horizon_event_probs': horizon_event_probs,
            'anomaly_score': anomaly_score
        }

    def _compute_event_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels_by_name: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute supervised event losses for multi-class and horizon binary heads.
        
        Applies configurable positive class weights to handle class imbalance
        for rare tsunami/volcano events per horizon.
        """
        event_losses = []
        device = outputs['event_logits'].device

        for label_name, labels in labels_by_name.items():
            if not isinstance(labels, torch.Tensor) or labels.numel() == 0:
                continue

            labels = labels.to(device)
            parts = label_name.split('_')
            horizon = parts[-1].replace('m', '') if parts else ''

            if label_name.startswith('event_'):
                event_losses.append(
                    F.cross_entropy(outputs['event_logits'], labels.long())
                )
            elif horizon in outputs['horizon_event_logits']:
                logits = outputs['horizon_event_logits'][horizon]
                if label_name.startswith('tsunami_'):
                    # Apply positive class weight for imbalanced tsunami labels
                    pos_weight = None
                    if self.pos_weights and horizon in self.pos_weights:
                        pos_weight = torch.tensor([self.pos_weights[horizon]], device=device)
                    event_losses.append(
                        F.binary_cross_entropy_with_logits(
                            logits[:, 0], labels.float(),
                            pos_weight=pos_weight
                        )
                    )
                elif label_name.startswith('volcano_'):
                    # Apply positive class weight for imbalanced volcano labels
                    pos_weight = None
                    if self.pos_weights and horizon in self.pos_weights:
                        pos_weight = torch.tensor([self.pos_weights[horizon]], device=device)
                    event_losses.append(
                        F.binary_cross_entropy_with_logits(
                            logits[:, 1], labels.float(),
                            pos_weight=pos_weight
                        )
                    )

        if not event_losses:
            return outputs['event_logits'].new_tensor(0.0)

        return torch.stack(event_losses).mean()
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, Dict],
        batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        x, y = batch
        
        # Forward pass
        outputs = self(x)
        
        # Reconstruction loss
        recon_loss = self.compute_reconstruction_loss(x, outputs['x_reconstructed'])
        
        # KL divergence
        kl_loss = self.compute_kl_divergence(outputs['mu'], outputs['logvar'])
        
        # Apply KL annealing
        beta = self.get_beta(self.global_step)
        
        # Total VAE loss
        vae_loss = recon_loss + beta * kl_loss
        
        # Event prediction loss (if labels available)
        event_loss = self._compute_event_loss(outputs, y) if y is not None else vae_loss.new_tensor(0.0)
        
        # Total loss
        total_loss = vae_loss + 0.1 * event_loss  # Weight event loss
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True)
        self.log('train_kl_loss', kl_loss, on_step=True, on_epoch=True)
        self.log('beta', beta, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, Dict],
        batch_idx: int
    ) -> Dict:
        """Validation step."""
        x, y = batch
        
        # Forward pass
        outputs = self(x)
        
        # Losses
        recon_loss = self.compute_reconstruction_loss(x, outputs['x_reconstructed'])
        kl_loss = self.compute_kl_divergence(outputs['mu'], outputs['logvar'])
        beta = self.get_beta(self.global_step)
        vae_loss = recon_loss + beta * kl_loss
        
        # Compute anomaly scores based on reconstruction error
        reconstruction_error = torch.mean((x - outputs['x_reconstructed']) ** 2, dim=(1, 2))
        
        # Log metrics with on_epoch=True for automatic aggregation in Lightning 2.x
        self.log('val_loss', vae_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True)
        
        return {
            'val_loss': vae_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'reconstruction_error': reconstruction_error,
            'anomaly_score': outputs['anomaly_score'],
            'event_probs': outputs['event_probs']
        }
    
    def on_validation_epoch_end(self) -> None:
        """Validation epoch end (Lightning 2.x compatible)."""
        # Store for analysis (metrics already logged via self.log with on_epoch=True)
        if hasattr(self, 'val_reconstruction_loss'):
            avg_recon = self.trainer.callback_metrics.get('val_recon_loss')
            avg_kl = self.trainer.callback_metrics.get('val_kl_loss')
            if avg_recon is not None:
                self.val_reconstruction_loss.append(avg_recon.item())
            if avg_kl is not None:
                self.val_kl_loss.append(avg_kl.item())
    
    def test_step(
        self,
        batch: Tuple[torch.Tensor, Dict],
        batch_idx: int
    ) -> Dict:
        """Test step."""
        x, y = batch
        outputs = self(x)
        
        # Compute reconstruction error per sample
        reconstruction_error = torch.mean((x - outputs['x_reconstructed']) ** 2, dim=(1, 2))
        
        return {
            'reconstruction_error': reconstruction_error,
            'anomaly_score': outputs['anomaly_score'],
            'event_probs': outputs['event_probs'],
            'latent': outputs['z'],
            'labels': y
        }
    
    def configure_optimizers(self) -> Dict:
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def compute_anomaly_score(
        self,
        x: torch.Tensor,
        threshold_percentile: float = 95
    ) -> torch.Tensor:
        """
        Compute anomaly score for input sequences.
        
        Combines reconstruction error with latent space probability.
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            
            # Reconstruction error
            recon_error = torch.mean((x - outputs['x_reconstructed']) ** 2, dim=(1, 2))
            
            # Normalize reconstruction error
            recon_score = recon_error / (recon_error.max() + 1e-8)
            
            # Combine with anomaly head output
            anomaly_score = 0.5 * recon_score + 0.5 * outputs['anomaly_score'].squeeze()
        
        return anomaly_score
    
    def predict_event(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict event probabilities.
        
        Returns:
            event_probs: Probabilities for each event type
            predictions: Predicted class
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            event_probs = outputs['event_probs']
            predictions = torch.argmax(event_probs, dim=-1)
        
        return event_probs, predictions
    
    def get_latent_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent space embedding for input."""
        self.eval()
        with torch.no_grad():
            mu, logvar, _ = self.encoder(x)
        return mu


class CryptoDataset(Dataset):
    """PyTorch Dataset for cryptocurrency time series."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[Dict[str, np.ndarray]] = None,
        transform=None
    ):
        self.X = torch.FloatTensor(X)
        self.y = {}
        
        if y is not None:
            for key, value in y.items():
                if isinstance(value, np.ndarray):
                    self.y[key] = torch.FloatTensor(value)
        
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        x = self.X[idx]
        
        if self.transform:
            x = self.transform(x)
        
        y = {key: val[idx] for key, val in self.y.items()} if self.y else {}
        
        return x, y


def create_dataloaders(
    X: np.ndarray,
    y: Dict[str, np.ndarray],
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders with time-series split."""
    
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # Time-series split (no shuffling!)
    train_dataset = CryptoDataset(X[:train_end], {k: v[:train_end] for k, v in y.items()})
    val_dataset = CryptoDataset(X[train_end:val_end], {k: v[train_end:val_end] for k, v in y.items()})
    test_dataset = CryptoDataset(X[val_end:], {k: v[val_end:] for k, v in y.items()})
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for time series
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="LSTM-VAE Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    model_config = config.get('model', {}).get('lstm_vae', {})
    vae_config = config.get('model', {}).get('vae', {})
    training_config = config.get('model', {}).get('training', {})
    
    # Create model
    model = LSTMVAE(
        input_dim=model_config.get('input_dim', 128),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_layers', 3),
        latent_dim=model_config.get('latent_dim', 64),
        dropout=model_config.get('dropout', 0.2),
        bidirectional=model_config.get('bidirectional', True),
        beta=vae_config.get('beta', 0.001),
        kl_annealing=vae_config.get('kl_annealing', True),
        kl_annealing_steps=vae_config.get('kl_annealing_steps', 10000),
        learning_rate=training_config.get('learning_rate', 0.001),
        weight_decay=training_config.get('weight_decay', 0.0001)
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(model)
