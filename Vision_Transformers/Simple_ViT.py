import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from vit_pytorch import ViT
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import matplotlib.pyplot as pltor
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from torch.cuda.amp import GradScaler, autocast


# Disable the DecompressionBombError by setting the MAX_IMAGE_PIXELS to None
Image.MAX_IMAGE_PIXELS = None

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to the .npz file.  npz was usefull for my project because of multiple files it maked organazation easier. although if you have ready the files for test validation and training its not necessery
npz_file_path = '/content/drive/MyDrive/Διπλωματική-ΑΣ/Code/Splits/grayscale_splits_train_test_val_split_346.npz'

# Check if the .npz file exists
if not os.path.exists(npz_file_path):
    print("NPZ file not found.")
    exit()  

# Load the .npz file
data = np.load(npz_file_path, allow_pickle=True)

# Retrieve datasets
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

print(f"Number of training samples: {len(X_train)}")
print(f"Number of validation samples: {len(X_val)}")
print(f"Number of test samples: {len(X_test)}")

# Convert file paths from numpy arrays to lists
X_train = X_train.tolist()
X_val = X_val.tolist()
X_test = X_test.tolist()


class CustomResizeAndPadOrCrop:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        width, height = img.size

        # Check if the image needs to be resized and padded or cropped
        if width < self.target_size[0] or height < self.target_size[1]:
            # Calculate padding to center the image
            padding_width = max((self.target_size[0] - width) // 2, 0)
            padding_height = max((self.target_size[1] - height) // 2, 0)

            # Create a new blank RGB image of target size
            padded_img = Image.new("L", self.target_size, color=0)  # Set color to black for RGB

            # Paste the original image onto the blank image at the center
            padded_img.paste(img, (padding_width, padding_height))

            # Return the padded image
            img = padded_img

        elif width > self.target_size[0] or height > self.target_size[1]:
            # Crop the image from the top-left corner
            img = transforms.functional.crop(img, 0, 0, self.target_size[1], self.target_size[0])

        return img


class EarlyStopping:
    def __init__(self, patience=25, verbose=True):
        self.patience = patience
        self.verbose = verbose  
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss






# Define transformations
transform = transforms.Compose([
    CustomResizeAndPadOrCrop((1024, 1024)),  # Adjust size as needed based on the model below,here it is this
    transforms.ToTensor(),
    
])

# Custom Dataset Class
class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Create datasets and dataloaders
batch_size = 16
train_dataset = ImageDataset(X_train, y_train, transform=transform)
val_dataset = ImageDataset(X_val, y_val, transform=transform)
test_dataset = ImageDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#Vision Transformer model
class CustomViT(nn.Module):
    def __init__(self):
        super(CustomViT, self).__init__()
        self.model = ViT(
            image_size=1024,
            patch_size=64,  # Adjusted patch size
            num_classes=2,
            dim=256,
            depth=12,
            heads=16,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
            channels=1
        )

    def forward(self, x):
        return self.model(x)

# Initialize the model
model = CustomViT()

# Ensure that GPU is available and being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adagrad(model.parameters(), lr=0.001)

# Define the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)

# Define early stopping
early_stopping = EarlyStopping(patience=5, verbose=True)

# Set up mixed precision training
scaler = GradScaler()

# Gradient accumulation steps
accumulation_steps = 4

# Train the model
epochs = 50
best_val_acc = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
        inputs, labels = inputs.to(device), labels.to(device)
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Validate the model
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    val_loss = val_loss / len(val_dataset)
    train_acc = correct_train / total_train
    val_acc = correct_val / total_val

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Step with the scheduler
    scheduler.step(val_loss)

    # Early stopping
    if early_stopping(val_loss, model):
        print("Early stopping")
        break

    # Save the model if validation accuracy is the best so far
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

# Load the best model over the epochs
model.load_state_dict(torch.load('best_model.pth'))


# Testing the model
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_loss /= len(test_dataset)
test_acc = test_correct / test_total

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Calculate Precision, Recall, and F1 Score
test_predictions = []
test_targets = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_predictions.extend(predicted.cpu().tolist())
        test_targets.extend(labels.cpu().tolist())

precision = precision_score(test_targets, test_predictions)
recall = recall_score(test_targets, test_predictions)
f1 = f1_score(test_targets, test_predictions)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Compute and print the confusion matrix
cm = confusion_matrix(test_targets, test_predictions)
print("\nConfusion Matrix:")
print("          Predicted")
print("         0      1")
print(f"Actual 0 {cm[0,0]}    {cm[0,1]}")
print(f"Actual 1 {cm[1,0]}    {cm[1,1]}")

# Save the trained model as .pth in the path you want
save_path = ''
torch.save(model.state_dict(), save_path)
print(f"Model saved at {save_path}")
