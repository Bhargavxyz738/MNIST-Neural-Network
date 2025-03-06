# Handwritten Digit Recognition with Neural Network 

[![License: MIT](LICENSE)

This repository contains code for training and testing a simple neural network to recognize handwritten digits using the MNIST dataset.  The project is implemented in PyTorch and demonstrates a basic image classification pipeline.  It includes both training and inference scripts, along with a pre-trained model.

## Project Structure

The project consists of the following files:

*   **`Training.py`**: This script contains the code for training the neural network. It defines the model architecture, data loading, training loop, and validation.
*   **`Inference.py`**: This script loads the pre-trained model and performs inference on the test dataset.  It includes functions to test a single image and visualize predictions with confidence scores.
*   **`image_recognisation.pth`**:  This file contains the saved weights of the pre-trained model.  It's loaded by `Inference.py`.
*   **`data/`**: This directory is where the MNIST dataset will be downloaded (created automatically by the scripts).

## Model Architecture

The model is a simple feedforward neural network (`SimpleNN`) with the following structure:

1.  **Input Layer**:  A fully connected layer (`nn.Linear`) that takes flattened 28x28 pixel images (784 input features) and outputs 128 features.
2.  **Hidden Layer**:  A ReLU activation function (`nn.ReLU`) applied to the output of the first layer.
3.  **Output Layer**:  A fully connected layer (`nn.Linear`) that takes the 128 features from the hidden layer and outputs 10 features (one for each digit class).
4.  **Softmax**: A Softmax activation function (`nn.Softmax(dim=1)`) is applied to the output layer to produce probabilities for each digit class.

```python
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
```

## Getting Started

### Prerequisites

*   **Python 3.6+**:  Ensure you have Python 3.6 or a later version installed.
*   **PyTorch**:  Install PyTorch following the instructions on the official PyTorch website ([https://pytorch.org/](https://pytorch.org/)).  Make sure to install the version appropriate for your system (CPU or GPU).
*   **torchvision**:  This package comes with PyTorch and contains the MNIST dataset and image transformations.
*   **Matplotlib**: Install using `pip install matplotlib`.
* **Numpy**: Install using `pip install numpy`

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Bhargavxyz738/NLP-number.git
    cd NLP-number
    ```

2.  **Install the required packages (if you haven't already):**

    ```bash
    pip install torch torchvision matplotlib numpy
    ```

### Usage

#### Training

To train the model, run the `Training.py` script:

```bash
python Training.py
```

*   This script will automatically download the MNIST dataset to the `./data` directory if it doesn't already exist.
*   The training process will run for 70 epochs (you can modify this in the script).
*   The script prints the training loss for each epoch, as well as the validation loss and accuracy.
*   The trained model's weights will be saved to a file named `image_recognisation.pth` in project root directory.

#### Inference

To perform inference using the pre-trained model, run the `Inference.py` script:

```bash
python Inference.py
```

*   This script loads the pre-trained model from `image_recognisation.pth`.
*   It then selects 5 random samples from the MNIST test dataset.
*   For each sample, it displays:
    *   The image of the handwritten digit.
    *   The model's prediction (P).
    *   The true label (T).
    *   The confidence score (C) of the prediction, expressed as a percentage.
* The output will be visualised using `matplotlib`.

#### Key Functions and Explanations

*   **`Training.py`**

    *   **Data Loading:**  Uses `torchvision.datasets.MNIST` to download and load the MNIST training and testing datasets.  `DataLoader` is used to create batches of data for training and validation.
    *   **Model Initialization:**  Creates an instance of the `SimpleNN` model and moves it to the appropriate device (CPU or GPU).
    *   **Loss Function and Optimizer:** Uses `nn.CrossEntropyLoss` as the loss function and `optim.Adam` as the optimizer.
    *   **Training Loop:** Iterates through the training data for a specified number of epochs.  For each batch:
        *   Performs a forward pass through the model.
        *   Calculates the loss.
        *   Performs a backward pass to compute gradients.
        *   Updates the model's weights using the optimizer.
        *   Prints training loss, validation loss and accuracy.
    *  **Model Saving**: The trained model's state dictionary is saved to `'image_recognisation.pth'`

*   **`Inference.py`**

    *   **Model Loading:** Loads the pre-trained model's state dictionary from `image_recognisation.pth`.
    *   **`test_single_image(image, model, device)`:**  Takes a single image, the model, and the device as input.  It performs a forward pass on the image and returns the predicted digit and the confidence score.
    *   **`visualize_predictions(test_dataset, model, device, num_samples=5)`:**  Selects a specified number of random samples from the test dataset.  For each sample, it calls `test_single_image` to get the prediction and confidence, and then uses `matplotlib` to display the image, prediction, true label, and confidence score.
    * **Device Agnostic:** Uses available divice (GPU if available else CPU).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

*   The MNIST dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
*   PyTorch documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
