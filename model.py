import torch
import torch.nn as nn

from transform import transform

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, class_names):
        super(SimpleCNN, self).__init__()

        self.transform = transform

        self.is_trained = False

        self.class_names = sorted(class_names)

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Reshape before fully connected layer
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def reset_weights(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def predict_class(self, image, device):
        input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

        self.eval()
        with torch.no_grad():
            model_output = self.forward(input_tensor.cuda())
        
        # Convert the output to probabilities using softmax
        probabilities = torch.nn.functional.softmax(model_output[0], dim=0)

        # Get the predicted class and associated probability
        predicted_class = torch.argmax(probabilities).item()
        predicted_probability = probabilities[predicted_class].item()

        predicted_class_name = self.class_names[predicted_class]

        probabilities_per_class = dict(zip(self.class_names, probabilities.tolist()))

        return predicted_class_name, predicted_probability, probabilities_per_class
