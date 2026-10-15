"""Flower NumPyClient implementation for local IoT federated training."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import flwr as fl

from src.federated.model import (
    IoTThreatClassifier,
    get_model_parameters,
    set_model_parameters,
)


class IoTFlowerClient(fl.client.NumPyClient):
    """
    Flower client simulating a local IoT device / gateway.

    Receives global model weights, trains on local private partition,
    and returns updated weights and local sample count to the server.
    """

    def __init__(
        self,
        client_id: int,
        model: IoTThreatClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        local_epochs: int = 2,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.model = model
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(device)

        self.model.to(self.device)

        # Convert local data to PyTorch Tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        self.dataset = TensorDataset(X_tensor, y_tensor)
        self.num_samples = len(X_train)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Return local model parameters as list of NumPy arrays."""
        return get_model_parameters(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train local model on private partition starting from received global parameters.
        """
        # 1. Update local model with current global parameters
        set_model_parameters(self.model, parameters)
        self.model.train()

        epochs = config.get("local_epochs", self.local_epochs)
        lr = config.get("learning_rate", self.learning_rate)
        batch_size = config.get("batch_size", self.batch_size)

        dataloader = DataLoader(
            self.dataset,
            batch_size=min(batch_size, max(1, self.num_samples)),
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Measure initial loss on received global parameters (before local optimization)
        self.model.eval()
        initial_loss = 0.0
        with torch.no_grad():
            eval_loader = DataLoader(
                self.dataset,
                batch_size=min(max(batch_size, 128), max(1, self.num_samples)),
                shuffle=False,
            )
            for bx, by in eval_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                initial_loss += criterion(self.model(bx), by).item() * bx.size(0)
        initial_loss = float(initial_loss / max(1, self.num_samples))
        self.model.train()

        epoch_losses = []
        correct = 0
        total = 0

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

            epoch_losses.append(running_loss / max(1, self.num_samples))

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        acc = float(correct / max(1, total)) if total > 0 else 0.0

        # Measure final loss on trained local parameters
        self.model.eval()
        final_loss = 0.0
        with torch.no_grad():
            for bx, by in eval_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                final_loss += criterion(self.model(bx), by).item() * bx.size(0)
        final_loss = float(final_loss / max(1, self.num_samples))

        loss_improvement = float(initial_loss - final_loss)

        metrics = {
            "client_id": self.client_id,
            "train_loss": round(avg_loss, 4),
            "train_acc": round(acc, 4),
            "num_samples": self.num_samples,
            "initial_loss": round(initial_loss, 4),
            "final_loss": round(final_loss, 4),
            "loss_improvement": round(loss_improvement, 4),
        }

        # Return updated parameters, sample count, and training metrics
        return get_model_parameters(self.model), self.num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate received model parameters on local partition.
        """
        set_model_parameters(self.model, parameters)
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)

                total_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        avg_loss = float(total_loss / max(1, total))
        acc = float(correct / max(1, total))

        return avg_loss, self.num_samples, {"local_acc": acc, "client_id": self.client_id}
