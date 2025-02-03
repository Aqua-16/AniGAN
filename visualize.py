import matplotlib.pyplot as plt
plt.style.use('dark_background')

def plot_model(epoch, gen_loss, disc_loss):
    epochs = list(range(1, epoch+1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, gen_loss, label="Generator Loss", marker='o', linestyle='-', color='#00BFFF')  # Cyan
    plt.plot(epochs, disc_loss, label="Discriminator Loss", marker='s', linestyle='--', color='#FF1493')  # Magenta

    # Labels and Title
    plt.xlabel("Epochs", color="white")
    plt.ylabel("Loss", color="white")
    plt.title("Training History", color="white")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
