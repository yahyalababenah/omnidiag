"""
OmniDiag — Federated Learning Aggregator
==========================================
Implements FedAvg (Federated Averaging) for privacy-preserving model updates
across multiple hospital sites. Each site trains locally and sends only the
model weight deltas (gradients) to the central server — raw patient data
never leaves the institution.

Architecture:
  - Flower (flwr) framework for federated coordination
  - FedAvg strategy: weighted average of client weight updates
  - Optional: differential privacy noise injection via opacus (if installed)

Usage (server startup):
    python -m backend.federated.aggregator --rounds 10 --min-clients 2

Usage (client per hospital site):
    python -m backend.federated.client --server-address <host>:8080 --disease heart_disease
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("omnidiag.federated")

# ---------------------------------------------------------------------------
# Flower server configuration
# ---------------------------------------------------------------------------

FL_SERVER_ADDRESS = os.getenv("FL_SERVER_ADDRESS", "0.0.0.0:8080")
FL_MIN_CLIENTS = int(os.getenv("FL_MIN_CLIENTS", "2"))
FL_MIN_FIT_CLIENTS = int(os.getenv("FL_MIN_FIT_CLIENTS", "2"))
FL_ROUNDS = int(os.getenv("FL_ROUNDS", "10"))


def _build_fedavg_strategy(initial_parameters=None):
    """Build a FedAvg strategy with conservative client requirements."""
    try:
        import flwr as fl  # type: ignore

        def weighted_average(metrics):
            accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
            examples = [num_examples for num_examples, _ in metrics]
            return {"accuracy": sum(accuracies) / max(sum(examples), 1)}

        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=FL_MIN_FIT_CLIENTS,
            min_evaluate_clients=FL_MIN_FIT_CLIENTS,
            min_available_clients=FL_MIN_CLIENTS,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=initial_parameters,
        )
        return strategy
    except ImportError:
        log.warning("flwr not installed — federated learning requires: pip install flwr")
        return None


def start_fl_server(rounds: int = FL_ROUNDS, server_address: str = FL_SERVER_ADDRESS):
    """
    Start the Flower federated learning server.

    Hospital clients connect to this server, send weight updates after local
    training, and receive the globally aggregated model.

    Args:
        rounds:         Number of federated training rounds.
        server_address: gRPC address to listen on (host:port).
    """
    try:
        import flwr as fl  # type: ignore

        strategy = _build_fedavg_strategy()
        if strategy is None:
            return

        log.info(f"Starting FL server at {server_address} for {rounds} rounds")
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )
    except ImportError:
        log.error("flwr not installed. Run: pip install flwr")
    except Exception as exc:
        log.error(f"FL server error: {exc!r}")


# ---------------------------------------------------------------------------
# Differential Privacy helper (opacus — optional)
# ---------------------------------------------------------------------------

def add_dp_noise(gradients: List[float], noise_multiplier: float = 1.1, max_grad_norm: float = 1.0) -> List[float]:
    """
    Apply Gaussian differential privacy noise to gradient updates.

    This clips each gradient to max_grad_norm and adds calibrated Gaussian
    noise scaled by noise_multiplier. Used before sending local updates
    to the aggregation server.

    Args:
        gradients:        Flat list of gradient values.
        noise_multiplier: Controls privacy-utility trade-off (higher = more private).
        max_grad_norm:    L2 clipping threshold.

    Returns:
        Noised gradient list.
    """
    try:
        import numpy as np  # type: ignore

        g = np.array(gradients, dtype=np.float32)
        norm = np.linalg.norm(g)
        if norm > max_grad_norm:
            g = g * (max_grad_norm / norm)
        noise = np.random.normal(0, noise_multiplier * max_grad_norm, size=g.shape)
        return (g + noise).tolist()
    except Exception as exc:
        log.warning(f"DP noise injection failed: {exc!r} — returning raw gradients")
        return gradients
