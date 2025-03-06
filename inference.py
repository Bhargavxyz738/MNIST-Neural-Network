import torch
import matplotlib.pyplot as plt
import random
from torchvision import datasets, transforms
import torch.nn as nn

# This is the model arcitecture which was used during training.
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
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SimpleNN().to(device)
model.load_state_dict(torch.load("image_recognisation.pth", map_location=device))
print("Loaded pre-trained model")
model.eval()

# Prepare the test dataset
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

def test_single_image(image, model, device):
    model.eval()
    with torch.no_grad():
       image = image.view(-1, 28*28).to(device)
       output = model(image)
       _, predicted = torch.max(output, 1)
       confidence = torch.max(output)
    return predicted.item(), confidence.item()
    
def visualize_predictions(test_dataset, model, device, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i in range(num_samples):

        random_index = random.randint(0, len(test_dataset) - 1)
        image, label = test_dataset[random_index]
        prediction, confidence  = test_single_image(image, model, device)
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f"P: {prediction}\nT: {label}\nC: {confidence*100:.2f}%")
        axes[i].axis('off')
    plt.show()
visualize_predictions(test_dataset, model, device)
