import matplotlib.pyplot as plt
import os
from PIL import Image

def save_images(images, num_images=8, title="Generated Images", save_dir="./generated_images", grid_filename="image_grid.png"):
    images = images[:num_images]

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save each individual image
    for i, img in enumerate(images):
        # Convert tensor to PIL image
        img_pil = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
        img_pil.save(os.path.join(save_dir, f'image_{i+1}.png'))

    # Save image grid
    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
    for i, img in enumerate(images):
        axes[i].imshow(img.permute(1, 2, 0).cpu().squeeze())
        axes[i].axis('off')

    plt.suptitle(title)
    plt.savefig(os.path.join(save_dir, grid_filename))  # Save the grid as an image
    plt.close(fig)  # Close the figure to free memory
