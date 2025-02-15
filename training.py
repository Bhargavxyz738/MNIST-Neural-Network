import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#import torch_xla.core.xla_model as xm

#device = xm.xla_device()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#This will be the neurone

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

"""
Step 3: Prepare the Data
"""
# Transform to normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training and testing data
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

"""
Step 4: Train the Network
"""
# Initialize the model, loss function, and optimizer
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 70

for epoch in range(epochs):
    model.train()
    #val_loss = 0.0

    running_loss = 0.0

    for images, labels in train_loader:
    # Move data to GPU
        images, labels = images.to(device), labels.to(device)


        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    #print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
    # Print training and validation loss
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader):.4f}")
    #print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(test_loader):.4f}")
    #print(f"Validation Accuracy: {100 * correct / total:.2f}%")




    correct = 0
    total = 0
    val_loss = 0.0

    model.eval()
    with torch.no_grad():
          for images, labels in test_loader:
                # Move data to GPU
               images, labels = images.to(device), labels.to(device)


               outputs = model(images)
         # Calculate the loss for validation set
               loss = criterion(outputs, labels)
               val_loss += loss.item()

               _, predicted = torch.max(outputs.data, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()
               print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(test_loader):.4f}")

# Print training and validation loss
   # print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader):.4f}")
       # print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(test_loader):.4f}")
               print(f"Validation Accuracy: {100 * correct / total:.2f}%")

