import torch
import optuna

from .models.resnet import create_resnet18
from .utils.data_loading import load_and_preprocess_data
from .utils.training import train_model
from .utils.evaluation import evaluate_model
from .utils.visualization import plot_training_history
from .utils.experiment_config import Experiment, ExperimentSettings


if __name__ == '__main__':
    num_epochs = 30
    # Функция objective для Optuna
    def objective(trial):
        # Значения для гиперпараметров
        lr = trial.suggest_float("learning_rate", 5e-5, 1e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)
        step_size = trial.suggest_int("step_size", 6, 9)
        gamma = trial.suggest_float("gamma", 0.07, 0.11)
        pretrained = trial.suggest_categorical("pretrained", [True])

        # Обновить ExperimentSettings с предложенными значениями
        settings = ExperimentSettings(
            experiment_name=f"OptunaTrial_{trial.number}",
            model_name="ResNet18",
            learning_rate=lr,
            batch_size=batch_size,
            num_epochs=num_epochs,
            weight_decay=weight_decay,
            step_size=step_size,
            gamma=gamma,
            pretrained=pretrained,
            random_seed=42
        )

        # Загрузить и предобработать данные
        trainloader, valloader, testloader = load_and_preprocess_data(settings.batch_size, settings.random_seed)

        # Создать модель
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = create_resnet18(settings.pretrained, device)

        # Обучить модель
        trained_model, history_df = train_model(model, trainloader, valloader, settings.learning_rate, settings.weight_decay, settings.step_size, settings.gamma, settings.num_epochs, device, settings.random_seed)

        # Оценить модель на валидационном наборе
        test_accuracy = evaluate_model(trained_model, testloader, device)
        # Вернуть метрику, которую Optuna будет оптимизировать (например, test_accuracy)
        return test_accuracy

    # Создать study (direction="maximize", чтобы максимизировать метрику)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner())

    # Запустить оптимизацию (n_trials - количество экспериментов)
    study.optimize(objective, n_trials=15)

    # Вывод результатов
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Получаем лучшие гиперпараметры из Optuna
    best_settings = ExperimentSettings(
        experiment_name="BestExperiment",
        model_name="ResNet18",
        learning_rate=trial.params['learning_rate'],
        batch_size=trial.params['batch_size'],
        num_epochs=num_epochs,
        weight_decay=trial.params['weight_decay'],
        step_size=trial.params['step_size'],
        gamma=trial.params['gamma'],
        pretrained=trial.params['pretrained'],
        random_seed=42
    )

    # Загружаем данные с лучшими гиперпараметрами
    trainloader, valloader, testloader = load_and_preprocess_data(best_settings.batch_size, best_settings.random_seed)

    # Создаем модель с лучшими гиперпараметрами
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model = create_resnet18(best_settings.pretrained, device)

    # Обучаем модель с лучшими гиперпараметрами
    trained_model, history_df = train_model(best_model, trainloader, valloader, best_settings.learning_rate, best_settings.weight_decay, best_settings.step_size, best_settings.gamma, best_settings.num_epochs, device, best_settings.random_seed)

    # Оцениваем модель на тестовом наборе
    test_accuracy = evaluate_model(trained_model, testloader, device)

    # Создаем объект Experiment
    experiment = Experiment(
        settings=best_settings,
        model=trained_model,
        history=history_df,
        test_accuracy=test_accuracy
    )

    # Сохраняем эксперимент
    experiment.save("./best_experiment")

    # Создаем графики обучения
    plot_training_history(history_df)
