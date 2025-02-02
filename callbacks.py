# TODO: Create callbacks to oversee progress of training
import torch
from torchvision.utils import save_image
import os

class VisualizeGeneratorCallback:
    def __init__(self, epoch, generator, batch_size = 64, noise_dim = 100, device = "cpu", save_dir = "sample_images"):
        self.show_at_epoch = epoch
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.generator = generator
        self.device = device
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def __call__(self, epoch):
        if((epoch+1)%self.show_at_epoch==0):
            self.generator.eval()
            with torch.no_grad():
                noise = torch.randn(self.batch_size, self.noise_dim, 1, 1, device=self.device)
                fake_images = self.generator(noise).to(self.device)
                fake_images = self.denormalize(fake_images)
                save_image(fake_images, f"{self.save_dir}/generated_images_batch_{epoch+1}.png", nrow=8, normalize=False)
            self.generator.train()

    def denormalize(tensor):
        return tensor * 0.5 + 0.5


class CheckpointCallback:
    def __init__(self, epoch, generator, discriminator, save_dir = "checkpoints"):
        self.save_at_epoch = epoch
        self.generator = generator
        self.discriminator = discriminator
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)
        
    def __call__(self, epoch):
        if((epoch+1)%self.save_at_epoch==0):
            torch.save(self.generator.state_dict(), f"{self.save_dir}/generator_epoch_{epoch+1}.pth")
            torch.save(self.discriminator.state_dict(), f"{self.save_dir}/discriminator_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")