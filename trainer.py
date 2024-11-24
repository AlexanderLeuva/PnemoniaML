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
from typing import Dict, Tuple, List, Any

# Import the custom dataset classes
from dataset_creator import CustomDataset, AugmentedBalancedDataset
from cnn import SimpleCNN, ImageMLP, ResNetCXR

from torch.utils.data import WeightedRandomSampler


"""
prevent moddel from hosrtcutting via just classifying hard labels -- it will always look towards the best optimal path. 
"""


# TODO: provide logging functionality so that I am better able to analyze / debug system
import pdb


from logger import add_performance_logging, setup_logger
import logging
logger = setup_logger(level = logging.DEBUG)



# Configure device and setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    """Collate function for DataLoader"""
    images = torch.stack([x['image'] for x in batch], axis=0)

    labels = torch.stack([x['label'] for x in batch], axis=0)
    
    ids = [x['id'] for x in batch]
    
    return {'ids': ids, "images": images, "labels": labels}




class TrainingConfig:
    """Configuration class for training hyperparameters"""
    def __init__(self):
        self.learning_rate = 0.000001 # lower learning rate bc there is so much more dataset ---> better gradient updates (5000 * 0.0001 = good updates), also here I learn that it is ( maximze deliberate steps and accuracy)
        self.batch_size = 16
        self.epochs = 5
        self.pos_weight =  1.0 # these weights are important in combinatgion to determine gradietn.
        self.neg_weight = 3.0
        self.drop_out = [0.3, 0.2, 0.1]

class MetricsTracker:
    """Class to track and compute various metrics during training, for one epoch"""
    def __init__(self, epoch = None):
        self.running_loss = 0.0
        self.correct = 0
        self.total = 0
        self.total_pos_pred = 0

        self.epoch = epoch

        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0


    def update(self, pred: torch.Tensor, y: torch.Tensor, loss: float):
        self.running_loss += loss
        predictions = pred.argmax(1)
        self.correct += (predictions == y).sum().item()
        self.total += y.size(0)

        
        self.TP += ((predictions == 1) & (y == 1)).sum().item()
        self.FP += ((predictions == 1) & (y == 0)).sum().item()
        self.TN += ((predictions == 0) & (y == 0)).sum().item()
        self.FN += ((predictions == 0) & (y == 1)).sum().item()


        # even when the values are 0 -- they both evalute to 1 - we MUST use AND because all of these just classify when they are both 0 or 1

    

        self.total_pos_pred += (predictions == 1).sum().item()

    def get_epoch_stats(self, num_batches: int) -> Dict[str, float]:

       
        return {
            'epoch': self.epoch,
            'loss': self.running_loss / num_batches,
            'accuracy': 100 * self.correct / self.total,
            'pos_pred_ratio': 100 * self.total_pos_pred / self.total,
            'recall - 1': self.TP / (self.TP + self.FN + 1e-8), 
            'precision - 1': self.TP / (self.TP + self.FP + 1e-8), 
            'recall - 0': self.TN / (self.TN + self.FP + 1e-8),
            'precision - 0': self.TN / (self.TN + self.FN +1e-8)
        }

class Trainer():
    def __init__(self, model, model_config):
        self.history = {'val': [], 'train': [], 'test': []}
        self.model = model
        self.config = model_config

        weights = torch.tensor([self.config.neg_weight, self.config.pos_weight]).to(DEVICE)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)     

        
    @add_performance_logging(logger, logging.INFO, type = 'train')
    def train_one_epoch(self, 
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train model for one epoch and return metrics"""
        self.model.train()
        metrics = MetricsTracker(epoch)

        difficult_cases = []
        
        for idx, batch in enumerate(dataloader):
            X = batch["images"].to(DEVICE)
            y = batch["labels"].to(DEVICE)
            
            # Forward pass and loss calculation
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            
            # Get all of the current images wrong.
            if epoch >= 4:
                predictions = pred.argmax(1)
                incorrect_mask = predictions != y
                if incorrect_mask.any():
                    difficult_cases.extend([{
                        'image': X[i],
                        'true_label': y[i].item(),
                        'predicted': predictions[i].item(),
                        'loss': loss.item()
                    } for i in range(len(y)) if incorrect_mask[i]])




            # Backpropagaion
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            metrics.update(pred, y, loss.item())

        if epoch >= 4 and difficult_cases:
            difficult_cases.sort(key=lambda x: x['loss'], reverse=True)
            num_to_keep = int(len(dataloader.dataset) * 0.15)
            return metrics.get_epoch_stats(len(dataloader)), difficult_cases[:num_to_keep]
            
            
            #if idx % 5 == 0:
        #     _log_batch_stats(epoch, total_epochs, idx, X, y, pred, loss, len(dataloader.dataset))
        
        return metrics.get_epoch_stats(len(dataloader)), None

    def train(self, train_dataloader, val_dataloader):

        difficult_cases = []
        # Training loop
        for epoch in range(self.config.epochs):
            metrics, hard_cases = self.train_one_epoch(train_dataloader, epoch)
            # Evaluation
            if hard_cases:
                difficult_cases = hard_cases 
            self.evaluate_model(val_dataloader)

 
    @add_performance_logging(logger, logging.INFO, type = 'val')
    def evaluate_model(self,
        dataloader: DataLoader,
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluate model and return test metrics and predictions"""
        self.model.eval()
        metrics = MetricsTracker()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch["images"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                
                pred = self.model(images)
                loss = self.loss_fn(pred, labels)
                
                probs = torch.softmax(pred, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                metrics.update(pred, labels, loss.item())
        

        return metrics.get_epoch_stats(len(dataloader)), None # this is messy and will change later for the hard cases.

    def calculate_metrics(self,
        probs: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate classification metrics at given threshold"""
        predictions = (probs >= threshold).astype(int)
        return {
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'class_0_precision': precision_score(labels == 0, predictions == 0),
            'class_0_recall': recall_score(labels == 0, predictions == 0),
            'class_1_precision': precision_score(labels == 1, predictions == 1),
            'class_1_recall': recall_score(labels == 1, predictions == 1)
        }

    def plot_metrics(self,
        probs: np.ndarray,
        labels: np.ndarray,
        thresholds: np.ndarray
    ) -> None:
        """Plot PR curve and calibration curve"""
        # PR curve
        precisions, recalls, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(recalls, precisions, label=f'PR curve (AP={ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
        
        plt.subplot(1, 2, 2)
        plt.plot(prob_pred, prob_true)
        plt.plot([0, 1], [0, 1], '--')
        plt.xlabel('Predicted probability')
        plt.ylabel('True probability')
        plt.title('Calibration curve')
        
        plt.tight_layout()
        plt.show()
    def evaluate_difficulties(self, model, dataloader):
        difficulties = {}
        model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                X = batch['images'].to(DEVICE)
                y = batch['labels'].to(DEVICE)
                indices = batch['ids']  # Original dataset indices
                
                pred = model(X)
                predictions = pred.argmax(1)
                incorrect_mask = predictions != y
                
                for i, idx in enumerate(indices):
                    difficulties[idx] = 3.0 if incorrect_mask[i] else 1
        
        return difficulties

    def evaluate_test_set(self, model, test_dataloader, device):
        """
        Evaluate model on test set and return predictions and true labels.
        
        Args:
            model: PyTorch model
            test_dataloader: DataLoader for test set
            device: Device to run evaluation on
        
        Returns:
            dict: Contains probabilities, true labels, and metrics
        """
        model.eval()
        test_probs = []
        test_labels = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                inputs, labels = batch['images'], batch['labels']
                


                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                test_probs.extend(probs.cpu().numpy()[:, 1])
                test_labels.extend(labels.cpu().numpy())
        
        test_probs = np.array(test_probs)
        test_labels = np.array(test_labels)
        
        # Calculate metrics
        test_metrics = self.calculate_metrics(test_probs, test_labels)
        
        return {
            'probabilities': test_probs,
            'true_labels': test_labels,
            'metrics': test_metrics
        }

def main():
    # Load configuration
    config = TrainingConfig()
    
    # Load datasets
    train_dataset = torch.load("data/trainset.pt")
    val_dataset = torch.load("data/valset.pt")
    test_dataset = torch.load("data/testset.pt")
    


    # Create dataloaders with proper collate_fn
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,  # Using the defined collate_fn
        shuffle=True
    )


    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        collate_fn=collate_fn,
        shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        collate_fn=collate_fn,
        shuffle=False
    )


    
    #pdb.set_trace()
    
    # Print dataset information
    
    # Initialize model
    """
    model = SimpleCNN(
        input_channels=1,
        num_classes=2,
        input_size=tuple(train_dataset[0][0].shape),
        dropout_rate=0.3
    ).to(DEVICE)

    """
    model = ResNetCXR( 
        num_classes=2,
        dropout_rate=config.drop_out).to(DEVICE)
    
    trainer = Trainer(model, config) 
    # Setup training
    trainer.train(train_dataloader, val_dataloader)

    return
    difficulties = trainer.evaluate_difficulties(model, train_dataloader)
    sampler = WeightedRandomSampler(
        [difficulties[x['id']] for x in train_dataset],
        num_samples=len(train_dataset)
    )
    weighted_loader = DataLoader(train_dataset, sampler=sampler, batch_size=config.batch_size, collate_fn=collate_fn)

    count1 = 0
    count0 = 0

    for i in range(len(difficulties)):
        if difficulties[i] == 3.0: 
            label = train_dataset[i]['label']

            if label == 1: 
                count1 += 1
            elif label == 0: 
                count0 += 1
            else: 
                raise Exception(f"No 1 or 0 values found for index {i}")


    # Define a new weighted CE loss for traaining on the
    weights = torch.tensor([count1 / (count1 + count0), count0 / (count0 + count1)]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate / 10)
    pdb.set_trace()    
    # After getting weighted_loader, start a new training loop
    epochs_with_hard = 10 # Number of additional epochs
    model.train()  # Ensure model is in training mode

    for epoch in range(epochs_with_hard):
    # Train with weighted sampling of hard cases
        metrics, hard_cases = train_one_epoch(weighted_loader, model, loss_fn, optimizer, epoch, config.epochs) 
        # Evaluate on validation set
        evaluate_model(val_dataloader, model, loss_fn)


 
    def plot_accuracies(metrics_list):
        # Extract epochs and accuracies
        val_accuracies = [epoch_dict['accuracy'] for epoch_dict in metrics_list['val']]
        train_accuracies = [epoch_dict['accuracy'] for epoch_dict in metrics_list['train']]
        epochs = range(len(train_accuracies))  # Assuming each entry is an epoch
        



        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        
        # Customize the plot
        plt.title('Training and Validation Accuracy Over Epochs', fontsize=14, pad=15)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        plt.xlim((0, len(train_accuracies)))  # Assuming each entry is an epoch))
        
        # Set y-axis to percentage scale
        plt.ylim([0, 100])  # Assuming accuracy is in percentage
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Show the plot
        plt.show()

        # Example usag

    plot_accuracies(history)
    

    torch.save(model, 'models/full_model.pt')

    #evaluate_model(test_dataloader, model, loss_fn)

    
    """
    plot_metrics(val_probs, val_labels, np.linspace(0, 1, 20))
    


    # TODO: can i find better way on communicating if I have made a better model? 
    # Calculate and display final metrics
    metrics = calculate_metrics(val_probs, val_labels)
    print("\nFinal Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.3f}")

    """


if __name__ == '__main__':
    main()