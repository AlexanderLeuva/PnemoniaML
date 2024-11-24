




import torch
from torch.utils.data import DataLoader


from dataset_creator import CustomDataset, AugmentedBalancedDataset
import torch.nn as nn

from cnn import SimpleCNN, ImageMLP


import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score

# VSCODE Shortcut of the day - Move a line up and down alt arrow key
# home and end - top and bottom 
# ctrl p - use this to navigate




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

def train_loop(dataloader, model, loss_fn, optimizer, epochs=20):
    size = len(dataloader.dataset)
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        total_pos_pred = 0  # Track number of positive predictions
        
        for idx, batch in enumerate(dataloader):
            X = batch["images"].to(device)
            y = batch["labels"].to(device)
            
            # Forward pass and loss calculation
            pred = model(X)
            loss = loss_fn(pred, y)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Statistics
            running_loss += loss.item()
            predictions = pred.argmax(1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
            total_pos_pred += (predictions == 1).sum().item()
            
            if idx % 5 == 0:
                current = idx * len(X)
                num_positive = (y == 1).sum().item()
                num_negative = len(y) - num_positive
                
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
                print(f"Batch stats - Positive: {num_positive}, Negative: {num_negative}")
                print(f"Predictions - Positive: {(predictions == 1).sum().item()}, Negative: {(predictions == 0).sum().item()}")
                print(f"Example preds vs actual: {predictions[:5].cpu().numpy()} vs {y[:5].cpu().numpy()}\n")
        
        # End of epoch statistics
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Accuracy: {epoch_acc:.2f}%")
        print(f"Positive predictions: {total_pos_pred} ({100 * total_pos_pred/total:.2f}%)\n")


def plot_pr_curve(precisions, recalls, average_precision):
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f'PR curve (AP={average_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

from sklearn.calibration import calibration_curve
def plot_calibration_curve(model, val_loader, device):
    probs = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            X = batch["images"].to(device)
            y = batch["labels"].to(device)
            
            outputs = model(X)
            prob = torch.softmax(outputs, dim=1)[:, 1]
            
            probs.extend(prob.cpu().numpy())
            labels.extend(y.cpu().numpy())
    
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
    
    plt.plot(prob_pred, prob_true)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration curve')
    plt.show()

#TODO: come up with a way of dynamically changing the threshold
def calculate_precision_recall(predictions, labels, threshold=0.5):
    # Convert probabilities to binary predictions
    predictions = (predictions >= threshold).float()
    
    # True Positives, False Positives, False Negatives
    tp = torch.sum((predictions == 1) & (labels == 1))
    fp = torch.sum((predictions == 1) & (labels == 0))
    fn = torch.sum((predictions == 0) & (labels == 1))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)
    
    return precision.item(), recall.item(), f1.item()


def test_loop(dataloader, model, loss_fn):
   # Set the model to evaluation mode - important for batch normalization and dropout layers
   # Unnecessary in this situation but added for best practices
   model.eval()
   size = len(dataloader.dataset)
   num_batches = len(dataloader)
   test_loss, correct = 0, 0

   all_preds = []
   all_labels = []

   all_probs = []


   # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
   # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
   with torch.no_grad():
       for batch in dataloader:




           images = batch["images"].to(device)
           labels = batch["labels"].to(device)


           pred = model(images)




           test_loss += loss_fn(pred, labels).item()






           #labels_argmax = labels.argmax(axis = 1)






           correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
           print(f"Example: {pred[0:5].argmax(1)}, {labels[0:5]}")



           probs = torch.softmax(pred, dim=1)[:, 1] # get the calss 1 probabilities 
           all_probs.extend(probs.cpu())


           all_preds.extend(pred.cpu().numpy())
           all_labels.extend(labels.cpu().numpy())



  


   predictions = np.array(all_preds).argmax(axis = 1)
   labels = np.array(all_labels)

   probs = np.array(all_probs)


   thresholds = torch.linspace(0, 1, 20)
   precisions = []
   recalls = []

   for threshold in thresholds:
       p, r, _ = calculate_precision_recall(torch.Tensor(probs), torch.Tensor(labels), threshold)
       precisions.append(p)
       recalls.append(r)

   print(thresholds)
   print(precisions)
   print(recalls)
   plot_pr_curve(precisions, recalls, sum(precisions) / len(precisions))





   test_loss /= num_batches
   correct /= size

    
# Calculate per-class metrics
   class_0_precision = precision_score(labels == 0, predictions == 0)
   class_0_recall = recall_score(labels == 0, predictions == 0)
   class_1_precision = precision_score(labels == 1, predictions == 1)
   class_1_recall = recall_score(labels == 1, predictions == 1)
    
   print(f"Class 0 - Precision: {class_0_precision:.3f}, Recall: {class_0_recall:.3f}")
   print(f"Class 1 - Precision: {class_1_precision:.3f}, Recall: {class_1_recall:.3f}")




   print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def collate_fn(batch):


   images = torch.stack([x[0] for x in batch], axis = 0)
   labels = torch.stack([x[1] for x in batch], axis = 0)


   return {"images": images, "labels": labels}




def main():

    learning_rate = 0.0005




    train_dataset = torch.load("data/trainset.pt")
    val_dataset = torch.load("data/valset.pt")
    test_dataset = torch.load("data/valset.pt")

    # Find a way to excessive document a very highi level picture of the process of evaluation and tasks for development
    train_dataloader = DataLoader(train_dataset, batch_size = 64, collate_fn = collate_fn, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 64, collate_fn = collate_fn, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = 64, collate_fn = collate_fn, shuffle = True)

    print(train_dataset[0][0].shape)
    print(val_dataset[0][0].shape)

    print(train_dataset[0])


    

    #model = SimpleCNN(input_channels=1, num_classes = 2, input_size = tuple(train_dataset[0][0].shape), dropout_rate= 0.1)
    model = ImageMLP(tuple(train_dataset[0][0].shape), hidden_sizes=[512, 256, 128], num_classes=2, dropout_rate=0.2)


    pos_weight = 1#0.001#1341/3875  # weight for positive class
    neg_weight = 1.0 


    weights = torch.tensor([neg_weight, pos_weight]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight = weights)



    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(val_dataloader, model, loss_fn)
    test_loop(test_dataloader, model, loss_fn)

if __name__ == '__main__':
    main()