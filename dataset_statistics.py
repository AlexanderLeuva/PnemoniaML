import torch
from dataset_creator import CustomDataset


train_dataset = torch.load("data/trainset.pt")
val_dataset = torch.load("data/valset.pt")
test_dataset = torch.load("data/valset.pt")


print(len(test_dataset))
print(len(val_dataset))
i = 50
print(val_dataset[i])
print(test_dataset[i])