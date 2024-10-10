import torch

def sample_images(model, num_samples=8, image_size=32, device='cuda'):
    model.eval()
    with torch.no_grad():
        # Sample from a normal distribution (latent space)
        z = torch.randn(num_samples, 48, image_size//4, image_size//4).to(device)
        
        # Invert the Glow model to get images back in pixel space
        generated_images = model.inverse(z)
        
        # Clamp the values to be between [0, 1] for valid image visualization
        generated_images = torch.clamp((generated_images + 1) / 2, min=0, max=1)
        
    return generated_images.cpu()
