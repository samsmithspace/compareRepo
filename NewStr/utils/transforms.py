from torchvision import transforms

'''
def get_transform():
    """
    Returns the image transformations to be applied to the dataset.

    Returns:
        torchvision.transforms.Compose: The composed image transformations.
    """
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(           # Normalize images using ImageNet means and stds
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform

'''


# In utils/transforms.py
def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Reduce to 224x224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        ),
    ])
    return transform