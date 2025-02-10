import dataclasses
import json
import os
from torch import save as torch_save


@dataclasses.dataclass
class ExperimentSettings:
    """Содержит все параметры, описывающие эксперимент."""
    experiment_name: str = "baseline"
    model_name: str = "ResNet34"
    learning_rate: float
    batch_size: int = 32
    num_epochs: int
    weight_decay: float
    step_size: int
    gamma: float
    random_seed: int = 42
    pretrained: bool = True

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Experiment:
    """Содержит информацию об одном запуске эксперимента."""
    settings: ExperimentSettings
    # Changed to object to avoid circular dependency
    model: object
    history: object
    test_accuracy: float

    def save(self, path):
        os.makedirs(path, exist_ok=True)  # Create directory if it doesn't exist

        settings_path = os.path.join(path, "settings.json")
        with open(settings_path, "w") as f:
            json.dump(self.settings.to_dict(), f, indent=4)

        model_path = os.path.join(path, "model.pth")
        torch_save(self.model.state_dict(), model_path)

        history_path = os.path.join(path, "history.csv")
        self.history.to_csv(history_path)

        print(f"Experiment saved to {path}")
