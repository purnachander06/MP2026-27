import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.federated.model import estimate_parameter_size_bytes

logger = logging.getLogger("FederatedStrategy")


def aggregate_fedavg(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    Standard sample-weighted FedAvg parameter aggregation:

        theta_global = sum(theta_k * n_k) / sum(n_k)

    Args:
        results: List of tuples where each tuple is (client_parameters, num_samples).

    Returns:
        Aggregated parameter tensors as a list of NumPy arrays.
    """
    if not results:
        raise ValueError("Cannot aggregate empty results in FedAvg.")

    total_samples = sum(num_samples for _, num_samples in results)
    if total_samples == 0:
        raise ValueError("Total sample count across participating clients cannot be zero.")

    # Initialize zero accumulators with shapes of first client's parameters
    first_client_params = results[0][0]
    weighted_weights = [np.zeros_like(layer, dtype=np.float64) for layer in first_client_params]

    for client_params, num_samples in results:
        weight = float(num_samples) / float(total_samples)
        for i, layer in enumerate(client_params):
            weighted_weights[i] += layer.astype(np.float64) * weight

    # Cast back to original dtype (float32)
    return [layer.astype(first_client_params[i].dtype) for i, layer in enumerate(weighted_weights)]


class LoggingFedAvg(FedAvg):
    """
    Flower FedAvg strategy extension with per-round metrics and communication tracking.
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[[int, Parameters, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        initial_parameters: Optional[Parameters] = None,
        **kwargs,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            **kwargs,
        )
        self.round_history: List[Dict[str, Any]] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using sample-weighted FedAvg and log communication payload."""
        if not results:
            return None, {}

        # Extract parameters and sample counts from Flower FitRes
        parsed_results = []
        for _, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            if not all(np.all(np.isfinite(p)) for p in params):
                logger.warning("Rejected client update containing non-finite parameters (NaN/Inf).")
                continue
            parsed_results.append((params, fit_res.num_examples))

        if not parsed_results:
            logger.warning("No valid finite client updates received in this round.")
            return None, {}

        # Perform sample-weighted aggregation
        aggregated_ndarrays = aggregate_fedavg(parsed_results)
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # Estimate communication payloads
        model_size_bytes = estimate_parameter_size_bytes(aggregated_ndarrays)
        num_successful_clients = len(results)
        num_failed_clients = len(failures)

        # Inbound payload: each client uploaded its model parameters
        upload_bytes_round = model_size_bytes * num_successful_clients
        # Outbound payload: server broadcasts global model to clients
        download_bytes_round = model_size_bytes * num_successful_clients

        round_record = {
            "round": server_round,
            "strategy": "FedAvg",
            "participating_clients": num_successful_clients,
            "failed_clients": num_failed_clients,
            "total_examples": sum(res.num_examples for _, res in results),
            "model_size_bytes": model_size_bytes,
            "upload_bytes_round": upload_bytes_round,
            "download_bytes_round": download_bytes_round,
        }
        self.round_history.append(round_record)

        return aggregated_parameters, {"participating_clients": num_successful_clients}


# -------------------------------------------------------------------------
# Standalone Mathematical Server Optimizer Updates for Unit Testing
# -------------------------------------------------------------------------

def calculate_pseudo_gradient(
    aggregated_weights: List[np.ndarray],
    current_weights: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Calculate the server pseudo-gradient (parameter displacement):
        Delta_t = bar{theta}^{(t)} - theta^{(t)}
    where bar{theta}^{(t)} is the sample-weighted average of client updates.
    """
    return [agg - cur for agg, cur in zip(aggregated_weights, current_weights)]


def server_adam_update(
    current_weights: List[np.ndarray],
    delta_t: List[np.ndarray],
    m_t: Optional[List[np.ndarray]],
    v_t: Optional[List[np.ndarray]],
    server_round: int,
    eta: float = 0.02,
    beta_1: float = 0.9,
    beta_2: float = 0.99,
    tau: float = 1e-3,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Compute one server FedAdam update step:
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * Delta_t
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * (Delta_t)^2
        eta_norm = eta * sqrt(1 - beta_2^{round + 1}) / (1 - beta_1^{round + 1})
        theta_{t+1} = theta_t + eta_norm * m_t / (sqrt(v_t) + tau)

    Returns:
        (new_weights, updated_m_t, updated_v_t)
    """
    if m_t is None or len(m_t) == 0:
        m_t = [np.zeros_like(x, dtype=np.float64) for x in delta_t]
    if v_t is None or len(v_t) == 0:
        v_t = [np.zeros_like(x, dtype=np.float64) for x in delta_t]

    new_m = [
        beta_1 * m + (1.0 - beta_1) * d.astype(np.float64)
        for m, d in zip(m_t, delta_t)
    ]
    new_v = [
        beta_2 * v + (1.0 - beta_2) * (d.astype(np.float64) ** 2)
        for v, d in zip(v_t, delta_t)
    ]

    # Bias correction factor
    round_idx = float(server_round)
    bias_m = 1.0 - np.power(beta_1, round_idx + 1.0)
    bias_v = 1.0 - np.power(beta_2, round_idx + 1.0)
    eta_norm = eta * (np.sqrt(bias_v) / (bias_m if bias_m > 0 else 1.0))

    new_weights = [
        (w.astype(np.float64) + eta_norm * m / (np.sqrt(v) + tau)).astype(w.dtype)
        for w, m, v in zip(current_weights, new_m, new_v)
    ]
    return new_weights, new_m, new_v


def server_yogi_update(
    current_weights: List[np.ndarray],
    delta_t: List[np.ndarray],
    m_t: Optional[List[np.ndarray]],
    v_t: Optional[List[np.ndarray]],
    server_round: int,
    eta: float = 0.02,
    beta_1: float = 0.9,
    beta_2: float = 0.99,
    tau: float = 1e-3,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Compute one server FedYogi update step:
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * Delta_t
        v_t = v_{t-1} - (1 - beta_2) * (Delta_t)^2 * sign(v_{t-1} - (Delta_t)^2)
        theta_{t+1} = theta_t + eta * m_t / (sqrt(v_t) + tau)
    """
    if m_t is None or len(m_t) == 0:
        m_t = [np.zeros_like(x, dtype=np.float64) for x in delta_t]
    if v_t is None or len(v_t) == 0:
        v_t = [np.zeros_like(x, dtype=np.float64) for x in delta_t]

    new_m = [
        beta_1 * m + (1.0 - beta_1) * d.astype(np.float64)
        for m, d in zip(m_t, delta_t)
    ]
    new_v = [
        v - (1.0 - beta_2) * (d.astype(np.float64) ** 2) * np.sign(v - (d.astype(np.float64) ** 2))
        for v, d in zip(v_t, delta_t)
    ]

    new_weights = [
        (w.astype(np.float64) + eta * m / (np.sqrt(np.maximum(v, 0.0)) + tau)).astype(w.dtype)
        for w, m, v in zip(current_weights, new_m, new_v)
    ]
    return new_weights, new_m, new_v


def server_adagrad_update(
    current_weights: List[np.ndarray],
    delta_t: List[np.ndarray],
    m_t: Optional[List[np.ndarray]],
    v_t: Optional[List[np.ndarray]],
    server_round: int,
    eta: float = 0.02,
    beta_1: float = 0.0,
    tau: float = 1e-3,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Compute one server FedAdagrad update step:
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * Delta_t
        v_t = v_{t-1} + (Delta_t)^2
        theta_{t+1} = theta_t + eta * m_t / (sqrt(v_t) + tau)
    """
    if m_t is None or len(m_t) == 0:
        m_t = [np.zeros_like(x, dtype=np.float64) for x in delta_t]
    if v_t is None or len(v_t) == 0:
        v_t = [np.zeros_like(x, dtype=np.float64) for x in delta_t]

    new_m = [
        beta_1 * m + (1.0 - beta_1) * d.astype(np.float64)
        for m, d in zip(m_t, delta_t)
    ]
    new_v = [
        v + (d.astype(np.float64) ** 2)
        for v, d in zip(v_t, delta_t)
    ]

    new_weights = [
        (w.astype(np.float64) + eta * m / (np.sqrt(np.maximum(v, 0.0)) + tau)).astype(w.dtype)
        for w, m, v in zip(current_weights, new_m, new_v)
    ]
    return new_weights, new_m, new_v


# -------------------------------------------------------------------------
# Flower FedOpt Strategy Subclasses with Communication & Metrics Logging
# -------------------------------------------------------------------------

from flwr.server.strategy import FedAdam, FedYogi, FedAdagrad


class LoggingFedAdam(FedAdam):
    """
    Flower FedAdam strategy extension with per-round metrics and communication tracking.
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        eta: float = 0.02,
        eta_l: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        accept_failures: bool = True,
        **kwargs,
    ):
        super().__init__(
            initial_parameters=initial_parameters,
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            accept_failures=accept_failures,
            **kwargs,
        )
        self.orig_dtypes = [p.dtype for p in parameters_to_ndarrays(initial_parameters)]
        self.round_history: List[Dict[str, Any]] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Perform FedAdam server aggregation and record communication statistics."""
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if parameters_aggregated is None:
            return None, {}

        # Preserve original precision (float32) and prevent float64 bloat
        self.current_weights = [w.astype(dt) for w, dt in zip(self.current_weights, self.orig_dtypes)]
        parameters_aggregated = ndarrays_to_parameters(self.current_weights)

        # Estimate communication payloads
        model_size_bytes = estimate_parameter_size_bytes(self.current_weights)
        num_successful_clients = len(results)
        num_failed_clients = len(failures)

        upload_bytes_round = model_size_bytes * num_successful_clients
        download_bytes_round = model_size_bytes * num_successful_clients

        round_record = {
            "round": server_round,
            "strategy": "FedAdam",
            "server_lr": self.eta,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "tau": self.tau,
            "participating_clients": num_successful_clients,
            "failed_clients": num_failed_clients,
            "total_examples": sum(res.num_examples for _, res in results),
            "model_size_bytes": model_size_bytes,
            "upload_bytes_round": upload_bytes_round,
            "download_bytes_round": download_bytes_round,
        }
        self.round_history.append(round_record)

        return parameters_aggregated, metrics_aggregated


class LoggingFedYogi(FedYogi):
    """
    Flower FedYogi strategy extension with per-round metrics and communication tracking.
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        eta: float = 0.02,
        eta_l: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        accept_failures: bool = True,
        **kwargs,
    ):
        super().__init__(
            initial_parameters=initial_parameters,
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            accept_failures=accept_failures,
            **kwargs,
        )
        self.orig_dtypes = [p.dtype for p in parameters_to_ndarrays(initial_parameters)]
        self.round_history: List[Dict[str, Any]] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if parameters_aggregated is None:
            return None, {}

        self.current_weights = [w.astype(dt) for w, dt in zip(self.current_weights, self.orig_dtypes)]
        parameters_aggregated = ndarrays_to_parameters(self.current_weights)

        model_size_bytes = estimate_parameter_size_bytes(self.current_weights)
        num_successful_clients = len(results)
        num_failed_clients = len(failures)

        upload_bytes_round = model_size_bytes * num_successful_clients
        download_bytes_round = model_size_bytes * num_successful_clients

        round_record = {
            "round": server_round,
            "strategy": "FedYogi",
            "server_lr": self.eta,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "tau": self.tau,
            "participating_clients": num_successful_clients,
            "failed_clients": num_failed_clients,
            "total_examples": sum(res.num_examples for _, res in results),
            "model_size_bytes": model_size_bytes,
            "upload_bytes_round": upload_bytes_round,
            "download_bytes_round": download_bytes_round,
        }
        self.round_history.append(round_record)

        return parameters_aggregated, metrics_aggregated


class LoggingFedAdagrad(FedAdagrad):
    """
    Flower FedAdagrad strategy extension with per-round metrics and communication tracking.
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        eta: float = 0.02,
        eta_l: float = 0.001,
        tau: float = 1e-3,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        accept_failures: bool = True,
        **kwargs,
    ):
        super().__init__(
            initial_parameters=initial_parameters,
            eta=eta,
            eta_l=eta_l,
            tau=tau,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            accept_failures=accept_failures,
            **kwargs,
        )
        self.orig_dtypes = [p.dtype for p in parameters_to_ndarrays(initial_parameters)]
        self.round_history: List[Dict[str, Any]] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if parameters_aggregated is None:
            return None, {}

        self.current_weights = [w.astype(dt) for w, dt in zip(self.current_weights, self.orig_dtypes)]
        parameters_aggregated = ndarrays_to_parameters(self.current_weights)

        model_size_bytes = estimate_parameter_size_bytes(self.current_weights)
        num_successful_clients = len(results)
        num_failed_clients = len(failures)

        upload_bytes_round = model_size_bytes * num_successful_clients
        download_bytes_round = model_size_bytes * num_successful_clients

        round_record = {
            "round": server_round,
            "strategy": "FedAdagrad",
            "server_lr": self.eta,
            "beta_1": self.beta_1,
            "tau": self.tau,
            "participating_clients": num_successful_clients,
            "failed_clients": num_failed_clients,
            "total_examples": sum(res.num_examples for _, res in results),
            "model_size_bytes": model_size_bytes,
            "upload_bytes_round": upload_bytes_round,
            "download_bytes_round": download_bytes_round,
        }
        self.round_history.append(round_record)

        return parameters_aggregated, metrics_aggregated


def create_strategy(
    strategy_name: str,
    initial_parameters: Parameters,
    server_learning_rate: float = 0.02,
    beta_1: float = 0.9,
    beta_2: float = 0.99,
    tau: float = 1e-3,
    accept_failures: bool = True,
    **kwargs,
):
    """
    Factory creating a configured strategy instance.
    """
    strat = strategy_name.lower().strip()
    gsfs_keys = {
        "aggregation_threshold",
        "perf_threshold",
        "grad_beta",
        "trigger_mode",
        "warmup_rounds",
        "max_staleness",
        "max_pool_size",
    }
    standard_kwargs = {k: v for k, v in kwargs.items() if k not in gsfs_keys}

    if strat in ("fedavg", "avg"):
        return LoggingFedAvg(
            initial_parameters=initial_parameters,
            **standard_kwargs,
        )
    elif strat in ("fedadam", "adam"):
        return LoggingFedAdam(
            initial_parameters=initial_parameters,
            eta=server_learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
            accept_failures=accept_failures,
            **standard_kwargs,
        )
    elif strat in ("fedyogi", "yogi"):
        return LoggingFedYogi(
            initial_parameters=initial_parameters,
            eta=server_learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
            accept_failures=accept_failures,
            **standard_kwargs,
        )
    elif strat in ("fedadagrad", "adagrad"):
        return LoggingFedAdagrad(
            initial_parameters=initial_parameters,
            eta=server_learning_rate,
            tau=tau,
            accept_failures=accept_failures,
            **standard_kwargs,
        )
    elif strat == "gsfs":
        from src.federated.gsfs_strategy import GSFSStrategy
        return GSFSStrategy(
            initial_parameters=initial_parameters,
            accept_failures=accept_failures,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown federated strategy: '{strategy_name}'. "
            f"Choose from: 'fedavg', 'fedadam', 'fedyogi', 'fedadagrad', 'gsfs'."
        )

