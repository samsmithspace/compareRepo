import torch


def evaluate(model, dataloader, device, mixed_precision=True):
    """Evaluate model accuracy with support for both traditional and ALaST models"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if mixed_precision and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    # Check if this is an ALaST model or a standard model
                    if hasattr(model, 'update_budgets'):  # It's an ALaST model
                        outputs = model(images, compute_deltas=False)
                    else:  # It's a standard model
                        outputs = model(images)
            else:
                # Check if this is an ALaST model or a standard model
                if hasattr(model, 'update_budgets'):  # It's an ALaST model
                    outputs = model(images, compute_deltas=False)
                else:  # It's a standard model
                    outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc