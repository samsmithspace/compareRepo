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
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB\n")

    # MODIFIED: Use CIFAR-100 specific normalization (matching trainbackup.py)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Just resizing
        transforms.ToTensor(),           # Converting to tensor
        transforms.Normalize(            # CIFAR-100 specific normalization
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),  # Just resizing
        transforms.ToTensor(),           # Converting to tensor
        transforms.Normalize(            # CIFAR-100 specific normalization
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
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

    # MODIFIED: Match batch size from trainbackup.py
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        #num_workers=2, pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        #num_workers=2, pin_memory=True if torch.cuda.is_available() else False
    )

    # Use test set as validation for CIFAR-100
    val_loader = test_loader

    # Common parameters
    num_epochs = 10  # Match trainbackup.py
    learning_rate = 1e-4
    # MODIFIED: Disable mixed precision to match trainbackup.py
    mixed_precision = False  # Changed from torch.cuda.is_available() to False

    # Number of training layers for ALaST
    n_train_layers = 8  # More layers for CIFAR-100 (100 classes)

    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  - Dataset: CIFAR-100")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Weight Decay: 0.1")
    print(f"  - Optimizer: AdamW (NO SCHEDULER)")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Number of Classes: {num_classes}")
    print(f"  - Mixed Precision: {mixed_precision}")
    print(f"  - ALaST Trainable Layers: {n_train_layers}")
    print(f"{'='*60}\n")

    try:
        # 1. Run traditional fine-tuning
        print("\n" + "="*60)
        print("STEP 1: RUNNING TRADITIONAL FINE-TUNING")
        print("="*60 + "\n")
        trad_start_time = time.time()
        traditional_model = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        
        # Print model info
        total_params = sum(p.numel() for p in traditional_model.parameters())
        trainable_params = sum(p.numel() for p in traditional_model.parameters() if p.requires_grad)
        print(f"Traditional Model:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}\n")
        
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
        os.makedirs('results', exist_ok=True)
        torch.save(traditional_model.state_dict(), 'results/traditional_cifar100.pth')
        print(f"\nTraditional model saved to results/traditional_cifar100.pth")

        # 2. Run ALaST fine-tuning
        print("\n" + "="*60)
        print("STEP 2: RUNNING ALAST FINE-TUNING")
        print("="*60 + "\n")
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
        print(f"\nALaST model saved to results/alast_cifar100.pth")

        # 3. Generate comparison plots
        print("\n" + "="*60)
        print("STEP 3: GENERATING COMPARISON PLOTS")
        print("="*60 + "\n")
        generate_comparison_plots(trad_metrics, alast_metrics, alast_model)

        # 4. Evaluate on test set
        print("\n" + "="*60)
        print("STEP 4: FINAL TEST SET EVALUATION")
        print("="*60 + "\n")
        traditional_test_acc = evaluate(traditional_model, test_loader, device, mixed_precision)
        alast_test_acc = evaluate(alast_model, test_loader, device, mixed_precision)

        print(f"Traditional Model - Test Accuracy: {traditional_test_acc:.2f}%")
        print(f"ALaST Model - Test Accuracy: {alast_test_acc:.2f}%")
        
        # Print comprehensive summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        
        print(f"\n--- ACCURACY COMPARISON ---")
        print(f"Traditional:")
        print(f"  Best Val Acc: {trad_best_acc:.2f}%")
        print(f"  Final Test Acc: {traditional_test_acc:.2f}%")
        print(f"\nALaST:")
        print(f"  Best Val Acc: {alast_best_acc:.2f}%")
        print(f"  Final Test Acc: {alast_test_acc:.2f}%")
        print(f"\nAccuracy Difference: {alast_test_acc - traditional_test_acc:+.2f}%")
        
        print(f"\n--- TRAINING TIME COMPARISON ---")
        print(f"Traditional Fine-tuning Time: {trad_total_time:.2f}s ({trad_total_time/60:.2f} minutes)")
        print(f"ALaST Fine-tuning Time: {alast_total_time:.2f}s ({alast_total_time/60:.2f} minutes)")
        print(f"Time Difference: {abs(trad_total_time - alast_total_time):.2f}s")
        
        if alast_total_time < trad_total_time:
            speedup = trad_total_time / alast_total_time
            time_saved = trad_total_time - alast_total_time
            print(f"ALaST was {((trad_total_time - alast_total_time) / trad_total_time * 100):.1f}% faster")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Time Saved: {time_saved:.2f}s ({time_saved/60:.2f} minutes)")
        else:
            slowdown = alast_total_time / trad_total_time
            time_lost = alast_total_time - trad_total_time
            print(f"Traditional was {((alast_total_time - trad_total_time) / alast_total_time * 100):.1f}% faster")
            print(f"Slowdown: {slowdown:.2f}x")
            print(f"Extra Time: {time_lost:.2f}s ({time_lost/60:.2f} minutes)")
        
        print(f"\n--- EFFICIENCY METRICS ---")
        avg_trad_time = np.mean(trad_metrics['time_per_epoch'])
        avg_alast_time = np.mean(alast_metrics['time_per_epoch'])
        print(f"Average Time per Epoch:")
        print(f"  Traditional: {avg_trad_time:.2f}s")
        print(f"  ALaST: {avg_alast_time:.2f}s")
        
        if 'peak_memory' in trad_metrics and trad_metrics['peak_memory']:
            avg_trad_mem = np.mean(trad_metrics['peak_memory'])
            avg_alast_mem = np.mean(alast_metrics['peak_memory'])
            print(f"\nAverage Peak Memory:")
            print(f"  Traditional: {avg_trad_mem:.2f}GB")
            print(f"  ALaST: {avg_alast_mem:.2f}GB")
            if avg_trad_mem > 0:
                mem_reduction = ((avg_trad_mem - avg_alast_mem) / avg_trad_mem * 100)
                print(f"  Memory {'Saved' if mem_reduction > 0 else 'Increase'}: {abs(mem_reduction):.1f}%")
        
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nAn error occurred during comparison: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run ALaST training on CIFAR-100 dataset"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB\n")

    # Create output directory
    os.makedirs('results', exist_ok=True)

    # MODIFIED: Use CIFAR-100 specific normalization (matching trainbackup.py)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Just resizing
        transforms.ToTensor(),           # Converting to tensor
        transforms.Normalize(            # CIFAR-100 specific normalization
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),  # Just resizing
        transforms.ToTensor(),           # Converting to tensor
        transforms.Normalize(            # CIFAR-100 specific normalization
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
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

    # MODIFIED: Match batch size from trainbackup.py
    batch_size = 64
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )

    # Use test set as validation (CIFAR-100 standard split)
    val_loader = test_loader

    # Create model
    print("\nCreating model...")
    print(f"Number of classes: {num_classes}")
    model = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    # Training parameters for CIFAR-100
    num_epochs = 30
    learning_rate = 1e-4
    # MODIFIED: Disable mixed precision to match trainbackup.py
    mixed_precision = False
    n_train_layers = 8  # More trainable layers for 100 classes

    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  - Method: ALaST")
    print(f"  - Dataset: CIFAR-100")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Weight Decay: 0.1")
    print(f"  - Optimizer: AdamW (NO SCHEDULER)")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Number of Classes: {num_classes}")
    print(f"  - Mixed Precision: {mixed_precision}")
    print(f"  - Trainable Layers: {n_train_layers}")
    print(f"{'='*60}\n")

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
        print(f"\nModel saved to results/alast_cifar100.pth")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

    # Plot training metrics
    plot_training_metrics(metrics, "cifar100")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Final validation accuracy: {metrics['val_acc'][-1]:.2f}%")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Average epoch time: {np.mean(metrics['time_per_epoch']):.2f}s")
    if 'peak_memory' in metrics and metrics['peak_memory']:
        print(f"Average peak memory: {np.mean(metrics['peak_memory']):.2f}GB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Choose which mode to run:
    # main()  # Just run ALaST training on CIFAR-100
    run_comparison()  # Run both traditional and ALaST training for comparison