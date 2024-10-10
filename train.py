import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import Glow
from invert import sample_images
from utils import save_images

# Instantiate and train the model
batch_size = 128
num_channels = 3  # CIFAR-10 images have 3 color channels
num_levels = 3    # Number of levels in Glow model
num_steps = 16    # Number of steps (ActNorm, 1x1 Conv, Affine Coupling) per level
epochs = 20000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = datasets.CIFAR10(root='cifar_data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = Glow(num_channels=num_channels, num_levels=num_levels, num_steps=num_steps).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(epochs):
    
    model.train()
    epoch_loss = 0
    for i, (x, _) in enumerate(tqdm(dataloader)):
        x = x.to(device)
        optimizer.zero_grad()

        z, log_det = model(x)
        prior_log_prob = -0.5 * torch.sum(z**2, dim=[1, 2, 3])
        loss = -torch.mean(prior_log_prob + log_det)
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    # Every visualize_every epochs, show generated images
    if (epoch + 1) % 1 == 0:
        sample_images_batch = sample_images(model, num_samples=8, image_size=32)
        # Assuming `generated_images` is a tensor of images
        save_images(sample_images_batch, num_images=8, save_dir="./output_images", grid_filename="generated_grid.png")
