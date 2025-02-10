# CNN-Based Image Classification using GoogLeNet, ResNet18, and VGG16

## Overview
This project explores the performance of different Convolutional Neural Network (CNN) architectures—GoogLeNet, ResNet18, and VGG16—on various image classification tasks using datasets such as MNIST, CIFAR10, and CIFAR100. The main objective was to evaluate the impact of hyperparameter tuning, data augmentation, and optimization techniques on model performance and optimization.

## Key Features
- **Multiple CNN Architectures:** GoogLeNet, ResNet18, and VGG16 were used.
- **Dataset Variety:** Experiments conducted on MNIST, CIFAR10, and CIFAR100.
- **Hyperparameter Tuning:** Investigated learning rates, batch sizes, optimizers, and training epochs.
- **Data Augmentation:** Applied horizontal flips, random rotations, and random cropping.
- **Training Optimization:** Experimented with SGD and Adam optimizers and cosine annealing learning rate schedulers.
- **Model Evaluation:** Compared accuracy, loss, and overfitting trends.

## Technology Stack
- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Data Handling:** NumPy, torchvision
- **Logging & Experiment Management:** `os`, `datetime`

```

## Installation & Setup
### Prerequisites
Ensure you have Python 3.8+ installed along with PyTorch and torchvision. You can install dependencies using:
```sh
pip install torch torchvision numpy matplotlib
```

### Clone Repository
```sh
git clone https://github.com/yourusername/cnn-classification.git
cd cnn-classification
```

### Dataset Preparation
Download and place the MNIST, CIFAR10, and CIFAR100 datasets in the `data/` directory. The datasets are automatically downloaded when running the script if not found.

## Model Training Pipeline
### Step 1: Data Preparation
- **Resize Images:** Images were resized to 224x224 to match input requirements of CNN models.
- **Normalization:** Standardized input values using dataset-specific means and standard deviations.
- **Data Augmentation:** Applied transformations such as flips, rotations, and random cropping.

```python
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])
```

### Step 2: Model Initialization
Each model was modified as needed to match dataset constraints (e.g., input channels for grayscale images, output classes for different datasets).

```python
model = googlenet(pretrained=False, aux_logits=True, init_weights=True)
model.fc = nn.Linear(1024, 10)  # Modify for CIFAR10 classification
```

### Step 3: Define Loss and Optimizer
Cross-entropy loss was used for classification, and different optimizers were tested:

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.006, momentum=0.9)
# Alternative: optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Step 4: Training Loop
The model was trained for multiple epochs while logging accuracy and loss at each step. Auxiliary classifiers were used in GoogLeNet to improve gradient flow.

```python
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Step 5: Model Evaluation
Performance was measured using accuracy and loss on validation and test sets.

```python
def evaluate_model(loader, model):
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    return 100 * correct / total, total_loss / len(loader)
```

## Results & Observations
### MNIST
- Models achieved over **99.5% accuracy** due to the dataset’s simplicity.
- Overfitting was observed, limiting further accuracy improvements.

### CIFAR10
- **GoogLeNet performed best**, reaching **92.04% accuracy** with data augmentation.
- **ResNet18 also performed well**, benefiting from batch normalization and skip connections.
- **VGG16 struggled due to high parameter count**, achieving **84.19% accuracy** at best.

### CIFAR100
- **ResNet18 outperformed other models** with **72.84% accuracy**, thanks to its deeper architecture and skip connections.
- **GoogLeNet followed closely** with **70.77% accuracy**, aided by its inception modules.
- **VGG16 was less effective**, maxing out at **66.99% accuracy** despite augmentation.

## Challenges & Optimizations
### Overfitting
- Applied **data augmentation** to increase dataset variability.
- Used **learning rate scheduling (cosine annealing)** to escape local minima.

### Model Convergence
- **SGD performed better than Adam** for most cases, except when fine-tuning larger models.
- **Increased batch sizes and epochs** helped improve generalization.

### Computational Complexity
- **VGG16 suffered from high memory consumption**, making it slower than ResNet18 and GoogLeNet.
- **GoogLeNet’s auxiliary classifiers improved gradient flow**, stabilizing training.



[mini project report.pdf](https://github.com/kianahs/DeepLearning-mini-project/files/15471550/mini.project.report.pdf)
