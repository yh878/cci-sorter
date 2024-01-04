import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim

from model import SimpleCNN
from transform import transform

def train(model, image_directory, num_epochs=5, lr=0.001, save_path="trained_model.pth"):
    batch_size = 4

    train_transform = transform

    dataset_train = ImageFolder(
        root=image_directory,
        transform=train_transform)
    
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the model to the device
    model.to(device)
    
    model.reset_weights()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0

        for inputs, labels in dataloader_train:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print epoch-wise loss
        average_loss = total_loss / len(dataloader_train)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Trained model saved to {save_path}")


if __name__ == "__main__":
    # Define the path to your dataset
    dataset_path = "SmallSampleDataset/sorted/"

    # Instantiate the model
    model = SimpleCNN(num_classes=5)

    # Train the model
    train(model, dataset_path, num_epochs=5, lr=0.001, save_path="trained_model.pth")
