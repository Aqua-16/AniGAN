import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import generator
import discriminator
import data
import callbacks as c
from alive_progress import alive_bar


class DCGAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(DCGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.g_loss_history = []
        self.d_loss_history = []

    def compile(self, g_optimizer, d_optimizer, loss_fn):

        self.g_optim = g_optimizer
        self.d_optim = d_optimizer
        self.loss_fn = loss_fn
    
    def train_step(self, real_images):

        batch_size = real_images.size(0)
        device = real_images.device
        noise_vector = torch.randn(batch_size, self.generator.noise_dim, 1, 1, device=device)

        # Training discriminator

        self.d_optim.zero_grad()

        pred_real = self.discriminator(real_images)
        real_labels = torch.ones(batch_size, 1, device=device).to(device)
        real_labels -= 0.05 * torch.rand_like(real_labels) # Found Error!!!
        assert pred_real.size() == real_labels.size(), f"Mismatch Shapes {pred_real.size()}\n{real_labels.size()}" 
        assert pred_real.device == real_labels.device, "Mismatch devices"
        d_loss_real = self.loss_fn(pred_real, real_labels)

        fake_images = self.generator(noise_vector).detach()
        fake_labels = torch.zeros(batch_size, 1, device=device).to(device)
        pred_fake = self.discriminator(fake_images)
        d_loss_fake = self.loss_fn(pred_fake, fake_labels)

        d_loss = (d_loss_real + d_loss_fake)/2
        d_loss.backward()

        self.d_optim.step()

        # Training generator

        self.g_optim.zero_grad()

        fake_images = self.generator(noise_vector)
        labels = torch.ones(batch_size, 1, device=device).to(device)
        pred = self.discriminator(fake_images)
        g_loss = self.loss_fn(pred, labels)

        g_loss.backward()
        self.g_optim.step()

        self.d_loss_history.append(d_loss.item())
        self.g_loss_history.append(g_loss.item())

        return {'d_loss': d_loss.item(), 'g_loss': g_loss.item()}
    
    def train(self, dataloader, epochs, callbacks=[]):
        n = len(dataloader)
        with alive_bar(n, title_length = 15, bar = 'blocks', spinner = 'vertical', dual_line = True) as bar:
            for epoch in range(epochs):
                bar.title(f"Epoch [{epoch+1}/{epochs}]")

                for real_images in dataloader:
                    real_images = real_images.to(device)
                    loss_dict = self.train_step(real_images)
                    bar.text(f"D Loss: {loss_dict['d_loss']:.4f}, G Loss: {loss_dict['g_loss']:.4f}")
                    bar()
                
                for callback in callbacks:
                    callback(epoch)
                if((epoch+1)!=epochs): 
                    bar(-n)
    
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    dataset = data.CustomDataset(root_dir = "..\images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    G = generator.Generator().to(device)
    D = discriminator.Discriminator().to(device)

    g_optimizer = optim.Adam(G.parameters(), lr = 0.0002, betas = [0.5, 0.999])
    d_optimizer = optim.Adam(D.parameters(), lr = 0.0002, betas = [0.5, 0.999])
    loss = nn.BCELoss()

    model = DCGAN(G, D)
    model.compile(g_optimizer, d_optimizer, loss)

    n = int(input("Epochs: "))
    if(n==0):
        real_images = next(iter(dataloader)).to(device)
        loss_dict = model.train_step(real_images)

        print(loss_dict)
    else:
        
        vis_callback = c.VisualizeGeneratorCallback(1, G, 64, device)
        save_callback = c.CheckpointCallback(1, G, D)
        model.train(dataloader, n, callbacks=[vis_callback, save_callback])