import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from timm import create_model
import time
import numpy as np
import json
from datasets import load_dataset

from train import train_alast, train_traditional_vit
from evaluate import evaluate
from utils import generate_comparison_plots, plot_training_metrics


class HuggingFaceDataset(torch.utils.data.Dataset):
    """Wrapper to convert Hugging Face dataset to PyTorch dataset"""

    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['img']  # PIL Image
        label = item['fine_label']  # CIFAR-100 has fine-grained labels

        if self.transform:
            image = self.transform(image)

        return image, label


def run_comparison():
    """Run both traditional and ALaST training for comparison"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Simple data transformations matching trainbackup.py (NO augmentation)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Just resizing
        transforms.ToTensor(),           # Converting to tensor
        transforms.Normalize(            # Normalization
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        ),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),  # Just resizing
        transforms.ToTensor(),           # Converting to tensor
        transforms.Normalize(            # Normalization
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        ),
    ])

    # Load CIFAR-100 dataset from Hugging Face
    print("Loading CIFAR-100 dataset from Hugging Face...")
    hf_dataset = load_dataset('uoft-cs/cifar100')

    # Create PyTorch datasets
    train_dataset = HuggingFaceDataset(hf_dataset['train'], transform=transform_train)
    test_dataset = HuggingFaceDataset(hf_dataset['test'], transform=transform_val)

    # CIFAR-100 has 100 classes
    num_classes = 100
    print(f"CIFAR-100 dataset loaded with {num_classes} classes")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    batch_size = 32  # Can use larger batch size for CIFAR-100
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        #num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False,
        #num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )

    # Use test set as validation for CIFAR-100 (or split train set if needed)
    val_loader = test_loader

    # Common parameters
    num_epochs = 10  # More epochs for CIFAR-100
    learning_rate = 1e-4  # Lower learning rate for stability
    mixed_precision = torch.cuda.is_available()

    # Number of training layers
    n_train_layers = 8  # More layers for CIFAR-100 (100 classes)

    try:
        # 1. Run traditional fine-tuning
        print("\n===== Running Traditional Fine-tuning =====")
        trad_start_time = time.time()
        traditional_model = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        traditional_model, trad_metrics, trad_best_acc = train_traditional_vit(
            model=traditional_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=learning_rate,
            device=device,
            mixed_precision=mixed_precision
        )
        trad_total_time = time.time() - trad_start_time

        # Save traditional model
        torch.save(traditional_model.state_dict(), 'results/traditional_cifar100.pth')

        # 2. Run ALaST fine-tuning
        print("\n===== Running Improved ALaST Fine-tuning =====")
        alast_start_time = time.time()
        alast_model_base = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        alast_model, alast_metrics, alast_best_acc = train_alast(
            model=alast_model_base,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=learning_rate,
            device=device,
            n_train_layers=n_train_layers,
            mixed_precision=mixed_precision
        )
        alast_total_time = time.time() - alast_start_time

        # Save ALaST model
        torch.save(alast_model.state_dict(), 'results/alast_cifar100.pth')

        # 3. Generate comparison plots
        print("\n===== Generating Comparison =====")
        generate_comparison_plots(trad_metrics, alast_metrics, alast_model)

        # 4. Evaluate on test set
        print("\n===== Evaluating on Test Set =====")
        traditional_test_acc = evaluate(traditional_model, test_loader, device, mixed_precision)
        alast_test_acc = evaluate(alast_model, test_loader, device, mixed_precision)

        print(f"Traditional Model - Test Accuracy: {traditional_test_acc:.2f}%")
        print(f"ALaST Model - Test Accuracy: {alast_test_acc:.2f}%")
        
        # Print training times
        print(f"\n===== Training Time Comparison =====")
        print(f"Traditional Fine-tuning Time: {trad_total_time:.2f}s ({trad_total_time/60:.2f} minutes)")
        print(f"ALaST Fine-tuning Time: {alast_total_time:.2f}s ({alast_total_time/60:.2f} minutes)")
        print(f"Time Difference: {abs(trad_total_time - alast_total_time):.2f}s")
        if alast_total_time < trad_total_time:
            print(f"ALaST was {((trad_total_time - alast_total_time) / trad_total_time * 100):.1f}% faster")
        else:
            print(f"Traditional was {((alast_total_time - trad_total_time) / alast_total_time * 100):.1f}% faster")

    except Exception as e:
        print(f"An error occurred during comparison: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run ALaST training on CIFAR-100 dataset"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs('results', exist_ok=True)

    # Simple data transformations matching trainbackup.py (NO augmentation)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Just resizing
        transforms.ToTensor(),           # Converting to tensor
        transforms.Normalize(            # Normalization
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        ),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),  # Just resizing
        transforms.ToTensor(),           # Converting to tensor
        transforms.Normalize(            # Normalization
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        ),
    ])

    # Load CIFAR-100 dataset from Hugging Face
    print("Loading CIFAR-100 dataset from Hugging Face...")
    hf_dataset = load_dataset('uoft-cs/cifar100')

    # Create PyTorch datasets
    train_dataset = HuggingFaceDataset(hf_dataset['train'], transform=transform_train)
    test_dataset = HuggingFaceDataset(hf_dataset['test'], transform=transform_val)

    # CIFAR-100 has 100 classes
    num_classes = 100
    print(f"CIFAR-100 dataset loaded with {num_classes} classes")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Prepare dataloaders
    batch_size = 64  # Can use larger batch size for CIFAR-100
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )

    # Use test set as validation (CIFAR-100 standard split)
    val_loader = test_loader

    # Create model
    print("Creating model...")
    print(f"Number of classes: {num_classes}")
    model = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    # Training parameters for CIFAR-100
    num_epochs = 30
    learning_rate = 1e-4
    mixed_precision = torch.cuda.is_available()
    n_train_layers = 8  # More trainable layers for 100 classes

    print(f"Starting improved ALaST fine-tuning for {num_epochs} epochs with lr={learning_rate}")

    # Train with ALaST
    alast_model, metrics, best_accuracy = train_alast(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=learning_rate,
        device=device,
        n_train_layers=n_train_layers,
        mixed_precision=mixed_precision
    )

    # Save trained model
    try:
        torch.save(alast_model.state_dict(), 'results/alast_cifar100.pth')
    except Exception as e:
        print(f"Error saving model: {str(e)}")

    # Plot training metrics
    plot_training_metrics(metrics, "cifar100")

    print(f"Training complete!")
    print(f"Final validation accuracy: {metrics['val_acc'][-1]:.2f}%")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Average epoch time: {np.mean(metrics['time_per_epoch']):.2f}s")
    if 'peak_memory' in metrics and metrics['peak_memory']:
        print(f"Average peak memory: {np.mean(metrics['peak_memory']):.2f}GB")


if __name__ == "__main__":
    # Choose which mode to run:
    #main()  # Just run ALaST training on CIFAR-100
    run_comparison()  # Run both traditional and ALaST training for comparison