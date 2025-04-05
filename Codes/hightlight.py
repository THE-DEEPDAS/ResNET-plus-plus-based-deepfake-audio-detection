import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math

# Import Res2Net Components
from torch.utils import model_zoo

model_urls = {
    'res2net101_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth',
}

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(scale - 1)])
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        if len(spx) != self.scale:
            raise ValueError(
                f"Unexpected split sizes: {len(spx)}, expected {self.scale}. "
                f"Width: {self.width}, Scale: {self.scale}, Input channels: {out.shape[1]}"
            )

        for i in range(self.scale - 1):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = torch.cat((out, sp), 1) if i > 0 else sp

        if self.scale != 1:
            out = torch.cat((out, spx[self.scale - 1]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size() != out.size():
            residual = F.interpolate(residual, size=out.size()[2:], mode='bilinear', align_corners=False)

        out += residual
        return self.relu(out)

class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=2):
        super(Res2Net, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))

def res2net101_26w_4s(pretrained=False, num_classes=2):
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], num_classes=num_classes)
    if pretrained:
        # Load the pre-trained weights
        state_dict = model_zoo.load_url(model_urls['res2net101_26w_4s'])
        
        # Remove `fc` layer weights from the state dictionary
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        
        # Load the remaining weights
        model.load_state_dict(state_dict, strict=False)
    return model

# Dataset and Preprocessing
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        audio, sr = librosa.load(file_path, sr=16000)
        audio = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr), ref=np.max)
        audio = cv2.cvtColor(cv2.normalize(audio, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        audio_hsv = cv2.cvtColor(audio, cv2.COLOR_BGR2HSV)
        image = Image.fromarray(audio_hsv)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# Highlight Forged Predictions
def highlight_forged_predictions(dataset, predictions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    forged_samples = [i for i, pred in enumerate(predictions) if pred == 0][:5]

    for idx in forged_samples:
        file_path = dataset.file_paths[idx]
        audio, sr = librosa.load(file_path, sr=16000)
        spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr), ref=np.max)
        spectrogram = cv2.normalize(spectrogram, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        spectrogram = cv2.applyColorMap(spectrogram, cv2.COLORMAP_JET)
        cv2.rectangle(spectrogram, (0, 0), (spectrogram.shape[1], spectrogram.shape[0]), (0, 0, 255), 10)
        output_path = os.path.join(output_dir, f"forged_{idx}.png")
        cv2.imwrite(output_path, spectrogram)

# Main Execution
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Example File Paths and Labels
    file_paths = ["path/to/audio1.wav", "path/to/audio2.wav"]  # Replace with actual paths
    labels = [1, 0]  # Replace with actual labels

    dataset = AudioDataset(file_paths, labels, transform)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained model
    model = res2net101_26w_4s(pretrained=True, num_classes=2)
    model = model.to(device)

    # Modify the fc layer to match the number of classes
    num_classes = 2  # Update this based on your task
    in_features = model.fc.in_features  # Get input features of the original fc layer
    model.fc = nn.Linear(in_features, num_classes)  # Replace the fc layer
    nn.init.xavier_uniform_(model.fc.weight)  # Initialize weights
    nn.init.constant_(model.fc.bias, 0)  # Initialize biases

    # Move the model to the device
    model = model.to(device)

    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)

    highlight_forged_predictions(dataset, predictions, output_dir="highlighted_samples")
