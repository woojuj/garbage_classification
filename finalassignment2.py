import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Set the device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the paths of the data from the course database
TRAIN_PATH = "/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Train"
VAL_PATH = "/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Val"
TEST_PATH = "/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Test"

# We create a dataset class 
class GarbageDataset(Dataset):
    def __init__(self, data_path, image_transform, tokenizer):
        self.image_paths = []
        self.texts = []
        self.labels = []
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        #Create the classes of the data in the folder
        self.label_map = {"Black": 0, "Blue": 1, "Green": 2, "TTR": 3}

        for label in os.listdir(data_path):
            label_path = os.path.join(data_path, label)
            if os.path.isdir(label_path) and label in self.label_map:
                for file_name in os.listdir(label_path):
                    if file_name.endswith(".jpg") or file_name.endswith(".png"):
                        image_path = os.path.join(label_path, file_name)
                        self.image_paths.append(image_path)
                        self.labels.append(self.label_map[label])

                        text_path = os.path.join(label_path, file_name.rsplit('.', 1)[0] + ".txt")
                        if os.path.exists(text_path):
                            with open(text_path, 'r') as f:
                                self.texts.append(f.read())
                        else:
                            self.texts.append("")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.image_transform(image)
        text = self.texts[idx]
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, encoding, label

# Multi-modal model
class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        # We start with a pretrained ResNet50 to extract the features of the images
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)
        # We use a Bert model to extract the features of the text
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_fc = nn.Linear(768, 512)
        # Combine the two model in a fully connected neural network, 
        self.fc1 = nn.Linear(1024, 512)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, image, text):
        img_features = self.resnet(image)
        text_outputs = self.bert(**text)
        text_features = self.bert_fc(text_outputs.pooler_output)
        
        # concatenate the results from the ResNet18 and Bert
        combined = torch.cat((img_features, text_features), dim=1)
        combined = self.fc1(combined)
        combined = self.batch_norm1(combined)
        combined = self.dropout1(combined)
        combined = self.fc2(combined)
        combined = self.batch_norm2(combined)
        combined = self.dropout2(combined)
        output = self.fc3(combined)
        return output

# Data transformations and tokenizer
# We perform a data augmentation owing to the overfitting in the training in previous attempts
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.RandomAffine(degrees=15, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Datasets and data loaders
train_dataset = GarbageDataset(TRAIN_PATH, image_transform, tokenizer)
val_dataset = GarbageDataset(VAL_PATH, image_transform, tokenizer)
test_dataset = GarbageDataset(TEST_PATH, image_transform, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model and move it to the device
model = MultiModalModel().to(device)

# Criterion, optimizer, and scheduler
# We set the optimizer and the scheduler to get a better accuracy, based on the knowledge learned in class.
class_counts = [len([label for label in train_dataset.labels if label == i]) for i in range(4)]
class_weights = torch.tensor([1.0 / count for count in class_counts], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4, step_size_up=5)

# Training function with print statements for debugging
# We put the number of epochs at 30, waiting that each epoch improve compared with the previous, but normally the model finish between the epoch 10 or 15
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30):
    early_stop_tolerance = 5
    best_val_accuracy = 0
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, texts, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            texts = {k: v.squeeze(1).to(device) for k, v in texts.items()}

            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        
        # To keep tracking the results and to determine how to change the parameters of the model, we print the accuracy of each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
        
        val_accuracy = evaluate_model(model, val_loader, validation=True)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_tolerance:
                print("Early stopping triggered.")
                break

# Evaluation function
# We evaluate the best model from the training in the dataset provided in class
def evaluate_model(model, dataloader, validation=False):
    model.eval()
    all_preds, all_labels = [], []
    incorrect_images, incorrect_labels, incorrect_preds = [], [], []

    with torch.no_grad():
        for images, texts, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            texts = {k: v.squeeze(1).to(device) for k, v in texts.items()}
            
            outputs = model(images, texts)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    incorrect_images.append(images[i].cpu())
                    incorrect_labels.append(labels[i].cpu().item())
                    incorrect_preds.append(preds[i].cpu().item())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {accuracy:.4f}")

    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('confusion_matrix.png')
    plt.close()

    class_labels = ["Black", "Blue", "Green", "TTR"]
    num_to_display = min(4, len(incorrect_images))
    fig, axes = plt.subplots(1, num_to_display, figsize=(15, 15))
    for i in range(num_to_display):
        img = incorrect_images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"True: {class_labels[incorrect_labels[i]]}\nPred: {class_labels[incorrect_preds[i]]}")
        axes[i].axis("off")
    # We create the image for the incorrect classification
    plt.suptitle("Incorrect Classifications")
    plt.savefig('incorrect_classifications.png')
    plt.close()
    return accuracy

# Plot ROC Curve function
def plot_roc_curve(model, dataloader, num_classes=4):
    model.eval()
    all_labels, all_probs = [], []
    
    with torch.no_grad():
        for images, texts, labels in dataloader:
            images, labels = images.to(device), labels.cpu().numpy()
            texts = {k: v.squeeze(1).to(device) for k, v in texts.items()}
            all_labels.extend(labels)
            probs = nn.Softmax(dim=1)(model(images, texts)).cpu().numpy()
            all_probs.extend(probs)

    all_labels = label_binarize(all_labels, classes=list(range(num_classes)))
    all_probs = np.array(all_probs)
    fpr, tpr, roc_auc, class_labels = {}, {}, {}, ["Black", "Blue", "Green", "TTR"]

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'{class_labels[i]} (AUC = {roc_auc[i]:.2f})')
    # We create the ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

# Train and evaluate
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
print("Training completed.")
model.load_state_dict(torch.load('best_model.pth'))
print("Model loaded.")
evaluate_model(model, test_loader)
print("Evaluation completed.")
plot_roc_curve(model, test_loader)
print("ROC curve plotted and saved.")

