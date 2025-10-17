# data/dataset.py
from datasets import load_dataset
import torch


def load_and_preprocess_dataset(transform):
    """
    Loads the Stanford Dogs dataset and applies preprocessing transformations.

    Args:
        transform (torchvision.transforms.Compose): The image transformations to apply.

    Returns:
        datasets.DatasetDict: The processed dataset dictionary.
    """
    # Load the Stanford Dogs dataset
    dataset = load_dataset("Alanox/stanford-dogs")
    
    print(dataset)
    print("Dataset columns before preprocessing:")
    for split in dataset:
        print(f"{split} columns: {dataset[split].column_names}")
    
    # Get all unique breed names and create a mapping to integers
    all_breeds = sorted(set(dataset['full']['target']))
    breed_to_idx = {breed: idx for idx, breed in enumerate(all_breeds)}
    idx_to_breed = {idx: breed for breed, idx in breed_to_idx.items()}
    
    num_classes = len(all_breeds)
    print(f"\nFound {num_classes} unique dog breeds")
    print(f"Sample breeds: {all_breeds[:5]}")
    
    # Split the 'full' dataset into train and test (80-20 split)
    dataset_split = dataset['full'].train_test_split(test_size=0.2, seed=42)
    
    # Create dataset_dict with train and test splits
    dataset_dict = {
        'train': dataset_split['train'],
        'test': dataset_split['test']
    }
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(dataset_dict['train'])} samples")
    print(f"  Test: {len(dataset_dict['test'])} samples")
    
    # Define a function to preprocess images
    def preprocess(example):
        # Convert image to RGB
        image = example['image'].convert('RGB')
        
        # Apply transformations
        pixel_values = transform(image)
        
        # Convert breed name to integer label
        label_idx = breed_to_idx[example['target']]
        
        # Return the preprocessed example
        return {
            'image': example['image'],  # Preserve the original image
            'pixel_values': pixel_values,
            'labels': label_idx  # Convert string label to integer
        }
    
    # Apply the preprocessing to train and test splits
    dataset_dict['train'] = dataset_dict['train'].map(preprocess)
    dataset_dict['test'] = dataset_dict['test'].map(preprocess)
    
    # Define the features to include
    columns = ['pixel_values', 'labels']
    
    # Set the format for PyTorch tensors
    dataset_dict['train'].set_format(type='torch', columns=columns)
    dataset_dict['test'].set_format(type='torch', columns=columns)
    
    # Store the mappings for later use (e.g., inference)
    dataset_dict['breed_to_idx'] = breed_to_idx
    dataset_dict['idx_to_breed'] = idx_to_breed
    dataset_dict['num_classes'] = num_classes
    
    return dataset_dict