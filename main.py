# TODO: Improve interface

from dcgan import DCGAN
from generator import Generator
from discriminator import Discriminator
import data
from callbacks import VisualizeGeneratorCallback, CheckpointCallback
from visualize import plot_model
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    dataset = data.CustomDataset(root_dir = "..\images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    print("Dataset has been loaded.")

    G = Generator().to(device)
    D = Discriminator().to(device)

    g_optimizer = optim.Adam(G.parameters(), lr = 0.0002, betas = [0.5, 0.999])
    d_optimizer = optim.Adam(D.parameters(), lr = 0.0002, betas = [0.5, 0.999])
    loss = nn.BCELoss()

    model = DCGAN(G, D)
    model.compile(g_optimizer, d_optimizer, loss)

    print("Model compilation successful.")

    if(input("Manually initialize weights? (y/n): ") == "y"):
        G.load_state_dict(input("Enter generator file path: "))
        D.load_state_dict(input("Enter discriminator file path: ")) # TODO make loading weights simpler
    
    epochs = int(input("Enter number of epochs: "))
    vis_callback = VisualizeGeneratorCallback(5, G, 64, device)
    save_callback = CheckpointCallback(5, G, D)
    model.train(dataloader, epochs, callbacks=[vis_callback, save_callback])

    plot_model(epochs, model.g_loss_history, model.d_loss_history)