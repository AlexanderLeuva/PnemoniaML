import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score
)
from sklearn.calibration import calibration_curve
import logging
from typing import Dict, Tuple, List, Any

# Import custom functions/classes from your original files
from dataset_creator import CustomDataset, AugmentedBalancedDataset
from cnn import SimpleCNN, ImageMLP, ResNetCXR
from logger import add_performance_logging, setup_logger

from trainer import collate_fn, evaluate_model

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_final_model():
    """Evaluate a trained model on the test dataset"""
    # Load the saved model
    model = torch.load('models/full_model.pt')
    model.eval()  # Set to evaluation mode
    
    # Load test dataset
    test_dataset = torch.load("data/testset.pt")
    
    # Create test dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,  # Using same batch size as training
        collate_fn=collate_fn,
        shuffle=False  # No need to shuffle test data
    )
    
    # Setup loss function with the same weights as training
    weights = torch.tensor([3.0, 1.0]).to(DEVICE)  # Using your original weights
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    
    # Evaluate model and get metrics
    test_metrics = evaluate_model(test_dataloader, model, loss_fn)
    
    # Get detailed predictions for additional analysis
    all_probs = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            images = batch["images"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Get probability of positive class
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Print detailed metrics
    print("\nTest Set Evaluation Results:")
    print("-" * 50)
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.2f}")
    
    # Plot PR curve and calibration curve
    plt.figure(figsize=(12, 5))
    
    # PR curve
    plt.subplot(1, 2, 1)
    precisions, recalls, _ = precision_recall_curve(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    plt.plot(recalls, precisions, label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Test Set)')
    plt.legend()
    plt.grid(True)
    
    # Calibration curve
    plt.subplot(1, 2, 2)
    prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], '--', label='Perfect calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve (Test Set)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return test_metrics, all_probs, all_labels

if __name__ == "__main__":
    # Setup logging
    logger = setup_logger(level=logging.INFO)
    logger.info("Starting model evaluation on test set")
    
    try:
        metrics, probs, labels = evaluate_final_model()
        
        # Calculate additional metrics at different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        print("\nMetrics at different thresholds:")
        print("-" * 50)
        for threshold in thresholds:
            metrics_at_threshold = calculate_metrics(probs, labels, threshold)
            print(f"\nThreshold: {threshold}")
            for metric_name, value in metrics_at_threshold.items():
                print(f"{metric_name}: {value:.3f}")
                
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise e