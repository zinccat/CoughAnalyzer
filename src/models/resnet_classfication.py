import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from multiprocessing import freeze_support


class CustomImageDataset(Dataset):
    def __init__(
        self, images_dir, labels_dir, transform=None, label_type="respiratory_condition"
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.label_type = label_type  # can be 'respiratory_condition', 'fever_muscle_pain', or 'status'
        self.image_files = sorted(
            [
                f
                for f in os.listdir(images_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        label_path = os.path.join(
            self.labels_dir, os.path.splitext(image_name)[0] + ".txt"
        )

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        try:
            with open(label_path, "r") as f:
                content = f.read().strip()

            # If the file is empty, set label to 0
            if content == "":
                label = 0
            else:
                parts = content.split(",")
                if self.label_type == "respiratory_condition":
                    # Use parts[0] to determine the label: 'False' -> 0, 'True' -> 1
                    if parts[0] == "False":
                        label = 0
                    elif parts[0] == "True":
                        label = 1
                    else:
                        raise ValueError(
                            f"Unknown respiratory condition label {parts[0]}."
                        )
                elif self.label_type == "fever_muscle_pain":
                    # Use parts[1] to determine the label: 'False' -> 0, 'True' -> 1
                    if parts[1] == "False":
                        label = 0
                    elif parts[1] == "True":
                        label = 1
                    else:
                        raise ValueError(f"Unknown fever/muscle pain label {parts[1]}.")
                elif self.label_type == "status":
                    # Use parts[2] to determine the label; for example:
                    # 'healthy' -> 0, 'symptomatic' -> 1, 'COVID-19' -> 2
                    status = parts[2]
                    if status == "healthy":
                        label = 0
                    elif status == "symptomatic":
                        label = 1
                    elif status in ["COVID-19", "COVID"]:
                        label = 2
                    else:
                        raise ValueError(f"Unknown status label {status}.")
                elif self.label_type == "gender":
                    # Use parts[3] to determine the label:
                    gender = parts[3]
                    if gender == "male":
                        label = 0
                    elif gender == "female":
                        label = 1
                    elif gender == "other":
                        label = 2
                    else:
                        raise ValueError(f"Unknown gender label {gender}.")
                else:
                    raise ValueError(f"Unknown label type {self.label_type}.")
        except Exception as e:
            print(f"Error reading label file {label_path}: {e}. Setting label=0.")
            label = 0

        return image, label


def train():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    label_type = "status"  #'gender' #'fever_muscle_pain' #'status'  # Change this to 'respiratory_condition' or 'fever_muscle_pain' as needed

    train_dataset = CustomImageDataset(
        images_dir="data/coughvid_images/train",
        labels_dir="data/coughvid_labels/train",
        transform=train_transform,
        label_type=label_type,
    )
    val_dataset = CustomImageDataset(
        images_dir="data/coughvid_images/val",
        labels_dir="data/coughvid_labels/val",
        transform=val_transform,
        label_type=label_type,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=8
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    num_classes = 3 if label_type == "status" or label_type == "gender" else 2

    dropout_rate = 0.0  # 2
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate), nn.Linear(num_features, num_classes)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # only train the last layer
    # optimizer = optim.AdamW(model.fc.parameters(), lr=3e-4, weight_decay=1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100.0 * correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss = val_loss / len(val_dataset)
        val_acc = 100.0 * val_correct / val_total

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    # torch.save(model.state_dict(), "resnet18_cough.pth")
    # print("Training complete. Model saved as resnet18_cough.pth.")


if __name__ == "__main__":
    freeze_support()
    train()
