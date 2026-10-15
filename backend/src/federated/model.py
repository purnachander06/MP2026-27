"""PyTorch neural network architecture and parameter conversion utilities for IoT threat detection."""

from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IoTThreatClassifier(nn.Module):
    """
    Lightweight feed-forward neural network for IoT binary threat detection.

    Designed for fast, reproducible CPU execution in federated simulation.
    Takes 125 preprocessed feature inputs and outputs 2-class logits (0: benign, 1: malicious).
    """

    def __init__(
        self,
        input_dim: int = 125,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout_rate: float = 0.15,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits."""
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute softmax probabilities [P(benign), P(malicious)]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extract model state_dict parameters as a list of NumPy arrays for Flower.
    """
    return [val.detach().cpu().clone().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Load model parameters from a list of NumPy arrays received from Flower FedAvg.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_parameter_size_bytes(parameters: List[np.ndarray]) -> int:
    """
    Estimate raw payload size in bytes for a set of model parameters (float32).
    """
    return sum(p.nbytes for p in parameters)
