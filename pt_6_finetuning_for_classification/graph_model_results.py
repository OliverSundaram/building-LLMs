import matplotlib.pyplot as plt

def plot_results(train_losses,
                 train_accs,
                 val_losses,
                 val_accs,
                 epochs):

    plt.figure()
    plt.plot(epochs, train_losses, c="red", label="Train losses")
    plt.plot(epochs, val_losses, c="blue", label="Val losses")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.show()

    plt.figure()
    plt.plot(epochs, train_accs, c="green", label="Train Accs")
    plt.plot(epochs, val_accs, c="yellow", label="Val Accs")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accs (%)")
    plt.show()