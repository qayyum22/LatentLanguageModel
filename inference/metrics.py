# Evaluation metrics for reasoning paths



def compute_accuracy(predictions, targets):
    correct = (predictions.argmax(dim=-1) == targets).sum().item()
    total = targets.numel()
    return correct / total