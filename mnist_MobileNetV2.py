# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Confirm GPU availability
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyper-parameters
num_epochs = 3
batch_size = 100
learning_rate = 0.001 

# Data transforms
# Required to make the input images compatible to ImageNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    
                         std=[0.229, 0.224, 0.225]) 
])

train_loader = DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST("./data", train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=False
)

# Load pretrained MobileNetV2
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze pretrained model
for param in model.features.parameters():
    param.requires_grad = False

# Replace the classifier to match MNIST classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

model = model.to(device)  # <-- Move model to GPU

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
#    for images, labels in train_loader:
    for i, (images, labels) in enumerate(train_loader):  
        images, labels = images.to(device), labels.to(device)  # <-- Move batch to GPU

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 100 == 0:
             print (f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}') 

    print(f"Epoch {epoch+1}: loss = {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100*correct/total:.2f}%")

