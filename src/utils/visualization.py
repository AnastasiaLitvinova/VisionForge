import matplotlib.pyplot as plt


def plot_training_history(history_df):
    """Создает и отображает графики обучения (loss и accuracy)."""
    # Создаем графики обучения
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # График loss
    axs[0].plot(history_df['loss'], label='Training Loss')
    axs[0].plot(history_df['val_loss'], label='Validation Loss')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # График accuracy
    axs[1].plot(history_df['accuracy'], label='Training Accuracy')
    axs[1].plot(history_df['val_accuracy'], label='Validation Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # Покажем графики
    plt.show()
