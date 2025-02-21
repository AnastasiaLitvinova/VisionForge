import torch
import pandas as pd
import dataclasses
import json
import os


@dataclasses.dataclass(frozen=True)
class ExperimentSettings:
    """Stores experiment settings."""
    experiment_name: str
    model_name: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    weight_decay: float
    num_workers: int
    patience: int
    device: torch.device

    @property
    def as_dict(self) -> dict:
        """Returns a dict representation of the class."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Experiment:
    """Stores information about a single experiment run."""
    settings: ExperimentSettings
    model_state_dict: dict
    training_history: pd.DataFrame
    test_accuracy: float

    def save(self, path: str) -> None:
        """Saves experiment data to a given path."""
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "settings.json"), "w") as f:
            json.dump(self.settings.as_dict, f, indent=4)

        torch.save(self.model_state_dict, os.path.join(path, "model.pth"))
        self.training_history.to_csv(
            os.path.join(path, "history.csv"), index=False)

        print(f"Saved experiment to {path}")
