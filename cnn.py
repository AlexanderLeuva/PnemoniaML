    


#plt.imshow(x_val[0])



# %%
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn




from typing import Tuple # Class for reinforcing output requriements
from torchvision import models 


# Model defining basic CNN block - for


class ImageMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], num_classes=2, dropout_rate=0.1):
        """
        Simple MLP for image classification
        
        Args:
            input_size: tuple of (channels, height, width) or just flattened dimension
            hidden_sizes: list of hidden layer sizes
            num_classes: number of output classes
            dropout_rate: dropout probability between layers
        """
        super().__init__()
        
        # Calculate flattened input size if tuple is provided
        if isinstance(input_size, tuple):
            self.flatten_size = input_size[0] * input_size[1] * input_size[2]
        else:
            self.flatten_size = input_size
            
        # Build layers dynamically
        layers = []
        
        # Input layer
        prev_size = self.flatten_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),  # Add batch normalization
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten the input if it's not already flat
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
            
        return self.network(x)
    
    def get_num_parameters(self):
        """Returns the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


"""
# Third convolutional block
nn.Conv2d(64, 128, kernel_size=3, padding=1),
nn.ReLU(),
nn.BatchNorm2d(128),
nn.MaxPool2d(2),
nn.Dropout(dropout_rate),


#TODO: Add 4th + 5th block to optimize decision making + summarization, 
nn.Conv2d(128, 256, kernel_size=3, padding=1),
nn.ReLU(),
nn.BatchNorm2d(256),
nn.MaxPool2d(2),
nn.Dropout(dropout_rate),

nn.Conv2d(256, 512, kernel_size=3, padding=1),
nn.ReLU(),
nn.BatchNorm2d(512),
nn.MaxPool2d(2),
nn.Dropout(dropout_rate),

"""
class SimpleCNN(nn.Module):
   """Basic CNN architecture for classification"""
   def __init__(self, input_channels: int, num_classes: int, input_size = (1, 200, 200), dropout_rate: float = 0.3):
       super(SimpleCNN, self).__init__()


       self.features = nn.Sequential(
           # First convolutional block
           nn.Conv2d(input_channels, 3, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.BatchNorm2d(3),
           nn.MaxPool2d(2),
           nn.Dropout(dropout_rate),

           # Second convolutional block
           nn.Conv2d(3,3, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.BatchNorm2d(3),
           nn.MaxPool2d(2),
           nn.Dropout(dropout_rate),

           # Third convolutional block
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),
       )


       # Calculate size of flattened features
       #self.feature_size = self._get_flat_size()




       # tuple defining for a single element the shape of the array.
       self.input_size = input_size




       self.feature_size = self._get_flat_size(self.input_size) # TODO: Figure out wayt o calculat ehte dimension rigorously.


       self.classifier = nn.Sequential(


           nn.Linear(self.feature_size, 10),
           nn.ReLU(),
           nn.Dropout(dropout_rate),
           nn.Linear(10, num_classes) # no softmax because torch automatically softmaxes logits
       )


   def _get_flat_size(self, input) -> int:
       # Helper function to calculate flattened size


       dummy_array = torch.zeros(self.input_size)
       dummy_array = dummy_array.unsqueeze(0)


       #print(dummy_array.shape)
      
       output = self.features(dummy_array)
       x = output.view(output.size(0), -1) # Flatten the 2d convolution map


       return x.shape[1] # return the first dimension of the x.shape


   def forward(self, x: torch.Tensor) -> torch.Tensor:
       x = self.features(x)




       # define the size of hte
       x = x.view(x.size(0), -1) # Flatten the 2d convolution map
       #print(f"{x.shape} this is the strat")


       x = self.classifier(x)
       return x





class ResNetCXR(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = [0.3, 0.2, 0.1]):
        super(ResNetCXR, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify first conv layer to accept grayscale
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final FC layer
        num_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # Custom classifier similar to your originali

        
        # when it comes to decisino making - is a feedforward network ideal? this seems more subject to noise
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate[0]),
            nn.Linear(num_features, 50),
            nn.ReLU(),
            nn.Dropout(dropout_rate[1]),
            nn.Linear(50, 20),
            nn.Dropout(dropout_rate[2]), 
            nn.Linear(20, num_classes)
        )
        
        # Initialize the new layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet features
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification
        x = self.classifier(x)
        return x

    def get_activation_maps(self):
        """Helper method for GradCAM later"""
        # Return the last convolutional layer
        return list(self.resnet.children())[-2]







