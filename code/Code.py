# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

# If using GPU, check for availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
class MalnutritionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MalnutritionCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

model = MalnutritionCNN().to(device)

# %%
from torchsummary import summary
# Now, print the model summary
summary(model, input_size=(3, 224,224))


# %%
# You can set the paths to your train and test data folders
train_data_path = "C:/Users/KUSHAL G/Downloads/Project 2nd  Sem/Computer Vision/Image_dataset/train/"
test_data_path = "C:/Users/KUSHAL G/Downloads/Project 2nd  Sem/Computer Vision/Image_dataset/val/"

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the train and test datasets
train_dataset = ImageFolder(train_data_path, transform=transform)
test_dataset = ImageFolder(test_data_path, transform=transform)

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %%
# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer (you can experiment with different optimizers and learning rates)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
num_epochs = 50
training_losses = []
training_accuracies = []

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = (correct_predictions / total_predictions) * 100

    training_losses.append(epoch_loss)
    training_accuracies.append(epoch_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

print("Training completed.")

# %%
model.eval()  # Set the model to evaluation mode
true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        true_labels.extend(labels.tolist())
        predicted_labels.extend(preds.tolist())

# Generate the classification report
class_names = ['Healthy', 'Unhealthy']  # Replace with your actual class names
from sklearn.metrics import classification_report,accuracy_score
report = classification_report(predicted_labels,true_labels)
accuracy = accuracy_score(true_labels,predicted_labels) * 100
print(f"Accuracy: {accuracy:.2f}%")
print(report)

# %%
import matplotlib.pyplot as plt

# Plotting training loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(training_losses, label="Training Loss")
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plotting training accuracy
plt.subplot(1, 2, 2)
plt.plot(training_accuracies, label="Training Accuracy", color="green")
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()

# %%
def test_single_image(image_path):
    model.eval()  # Set the model to evaluation mode

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    print(prediction)

    if prediction == 0:
        print("The image represents a healthy child.")
    else:
        print("The image represents a child with malnutrition.")

# Test your single image
image_path = "/content/drive/MyDrive/Dataset/testing_img/image_726.jpg"
test_single_image(image_path)

# %%



