import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

from models.vit_model import get_modified_vit_model


def get_cifar100_transform():
    """Get transforms for CIFAR100 dataset"""
    return transforms.Compose([
        transforms.Resize((224, 224)),  # ViT expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                           std=[0.2675, 0.2565, 0.2761])
    ])


def load_cifar100_dataset(transform):
    """Load CIFAR100 dataset"""
    train_dataset = datasets.CIFAR100(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.CIFAR100(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    return {
        'train': train_dataset,
        'test': test_dataset,
        'num_classes': 100,
        'class_names': train_dataset.classes
    }


def train_model(model, dataset_dict, dataloaders, criterion, optimizer, device, num_epochs=5):
    """
    Trains the model and evaluates on the validation set.
    """
    steps_per_epoch = len(dataloaders['train'])
    total_steps = steps_per_epoch * num_epochs
    print(f"Total steps: {total_steps}")

    # Start timing for total training
    total_training_start_time = time.time()

    step = 0  # Initialize step counter

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        epoch_start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs)

            # Compute the loss
            loss = criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            step += 1  # Increment step counter

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        print(f'Training Loss: {epoch_loss:.4f}')

        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                logits = model(inputs)

                # Get predictions
                _, preds = torch.max(logits, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_accuracy = correct / total
        epoch_time = time.time() - epoch_start_time
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        print(f'Epoch Time: {epoch_time:.2f} seconds')

    # Calculate total training time
    total_training_time = time.time() - total_training_start_time
    print(f'\n{"=" * 60}')
    print(f'Total Fine-tuning Time: {total_training_time:.2f} seconds ({total_training_time / 60:.2f} minutes)')
    print(f'Average Time per Epoch: {total_training_time / num_epochs:.2f} seconds')
    print(f'{"=" * 60}\n')

    return model


def test_model(model, dataloader, device):
    """
    Evaluates the model on the test set.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(inputs)

            # Get predictions
            _, preds = torch.max(logits, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')
    return test_accuracy


def main():
    # Start total script time
    script_start_time = time.time()

    # Force CUDA usage - will raise error if CUDA is not available
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available! Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support installed.")

    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # Load and preprocess the dataset
    transform = get_cifar100_transform()
    dataset_dict = load_cifar100_dataset(transform)

    # Get number of classes from the dataset
    num_classes = dataset_dict['num_classes']
    
    print(f"\nDetected {num_classes} CIFAR100 classes")
    print(f"Example classes: {dataset_dict['class_names'][:5]}")

    # Create DataLoaders
    batch_size = 32
    dataloaders = {
        'train': DataLoader(dataset_dict['train'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(dataset_dict['test'], batch_size=batch_size)
    }

    # Initialize the model with the detected number of classes
    model = get_modified_vit_model(num_classes)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    #optimizer = optim.AdamW(model.parameters(), lr=1e-3)


    # Training parameters
    num_epochs = 10
    print(f"\nTraining Configuration:")
    print(f"  - Dataset: CIFAR100")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning Rate: 1e-4")
    print(f"  - Weight Decay: 0.1")
    print(f"  - Optimizer: AdamW")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Number of Classes: {num_classes}\n")

    # Train the model
    model = train_model(model, dataset_dict, dataloaders, criterion, optimizer, device, num_epochs=num_epochs)

    # Test the model
    print("\n===== Final Test Evaluation =====")
    test_accuracy = test_model(model, dataloaders['test'], device)

    # Total script execution time
    total_script_time = time.time() - script_start_time
    print(f'\n{"=" * 60}')
    print(f'FINAL SUMMARY:')
    print(f'{"=" * 60}')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    print(f'Total Script Execution Time: {total_script_time:.2f} seconds ({total_script_time / 60:.2f} minutes)')
    print(f'{"=" * 60}\n')

    # Save the model (optional)
    torch.save(model.state_dict(), 'vit_cifar100_finetuned.pth')


if __name__ == '__main__':
    main()