"""
OmniDiag — Federated Learning Client
======================================
Implements a Flower client that runs at each hospital site.
The client:
  1. Loads local patient data (never leaves the site)
  2. Trains the OmniDiag XGBoost model locally for a few iterations
  3. Sends only the weight delta to the FL server
  4. Receives the globally aggregated model

Run at each hospital:
    python -m backend.federated.client \
        --server-address <central-server>:8080 \
        --disease heart_disease \
        --data-path /local/patient_data.csv
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

log = logging.getLogger("omnidiag.federated.client")


class OmniDiagFLClient:
    """
    Flower client wrapper for OmniDiag.

    Each hospital runs one instance of this client. Local training happens
    entirely on local data; only serialised model parameters are transmitted.
    """

    def __init__(self, disease: str, data_path: str, model_path: str):
        self.disease = disease
        self.data_path = data_path
        self.model_path = model_path
        self._model = None
        self._X = None
        self._y = None

    def _load_data(self):
        if self._X is not None:
            return
        try:
            import pandas as pd  # type: ignore
            df = pd.read_csv(self.data_path)
            target = df.columns[-1]
            self._X = df.drop(columns=[target]).values
            self._y = df[target].values
            log.info(f"Loaded {len(df)} local records for {self.disease}")
        except Exception as exc:
            log.error(f"Failed to load local data: {exc!r}")
            raise

    def _load_model(self):
        if self._model is not None:
            return
        try:
            with open(self.model_path, "rb") as f:
                self._model = pickle.load(f)
        except Exception as exc:
            log.error(f"Failed to load model from {self.model_path}: {exc!r}")
            raise

    def get_parameters(self) -> List[bytes]:
        """Serialise model parameters for transmission to the server."""
        self._load_model()
        return [pickle.dumps(self._model)]

    def set_parameters(self, parameters: List[bytes]):
        """Deserialise and apply aggregated parameters from the server."""
        if parameters:
            self._model = pickle.loads(parameters[0])

    def fit(self, parameters: List[bytes], config: Dict) -> Tuple[List[bytes], int, Dict]:
        """Local training round."""
        self.set_parameters(parameters)
        self._load_data()
        self._load_model()
        try:
            # XGBoost incremental update (continue training from current model state)
            import xgboost as xgb  # type: ignore
            dtrain = xgb.DMatrix(self._X, label=self._y)
            self._model = xgb.train(
                params={"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 5},
                dtrain=dtrain,
                num_boost_round=10,
                xgb_model=self._model,
                verbose_eval=False,
            )
            n = len(self._y)
            return self.get_parameters(), n, {"dataset_size": n}
        except Exception as exc:
            log.error(f"Local training failed: {exc!r}")
            return parameters, 0, {}

    def evaluate(self, parameters: List[bytes], config: Dict) -> Tuple[float, int, Dict]:
        """Local evaluation round."""
        self.set_parameters(parameters)
        self._load_data()
        try:
            import xgboost as xgb  # type: ignore
            import numpy as np  # type: ignore
            dtest = xgb.DMatrix(self._X, label=self._y)
            preds = self._model.predict(dtest)
            labels = (preds > 0.5).astype(int)
            accuracy = float(np.mean(labels == self._y))
            return 0.0, len(self._y), {"accuracy": accuracy}
        except Exception as exc:
            log.error(f"Local evaluation failed: {exc!r}")
            return 0.0, 0, {}


def start_fl_client(
    disease: str,
    data_path: str,
    model_path: str,
    server_address: str = "localhost:8080",
):
    """Start a Flower client connecting to the FL server."""
    try:
        import flwr as fl  # type: ignore

        raw_client = OmniDiagFLClient(disease, data_path, model_path)

        class _FlowerClient(fl.client.NumPyClient):
            def get_parameters(self, config):
                return raw_client.get_parameters()

            def fit(self, parameters, config):
                return raw_client.fit(parameters, config)

            def evaluate(self, parameters, config):
                return raw_client.evaluate(parameters, config)

        log.info(f"Connecting FL client to {server_address}")
        fl.client.start_numpy_client(server_address=server_address, client=_FlowerClient())
    except ImportError:
        log.error("flwr not installed. Run: pip install flwr")
    except Exception as exc:
        log.error(f"FL client error: {exc!r}")
