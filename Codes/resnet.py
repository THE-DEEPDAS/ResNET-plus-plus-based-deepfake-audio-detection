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
import copy  # Add this import at the top with other imports
import torchaudio
import torch.nn.functional as F  # Add this import

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
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, stride=stride, bias=False)  # Set stride=stride
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)  # Keep stride=1
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
    """Initialize Res2Net model with optional pretrained weights and custom number of classes"""
    model = Res2Net(Bottle2neck, [3, 4, 23, 3])
    
    if pretrained:
        # Load pretrained weights
        state_dict = model_zoo.load_url(model_urls['res2net101_26w_4s'])
        
        # Remove the final fully connected layer weights from pretrained model
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        
        # Load the modified state dict
        model.load_state_dict(state_dict, strict=False)
        
        # Replace the final fully connected layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Initialize the new layer
        nn.init.xavier_uniform_(model.fc.weight)
        nn.init.zeros_(model.fc.bias)
    
    return model

# Dataset and Preprocessing
class AudioDataset(Dataset):
    def __init__(self, base_path, protocol_path, transform=None):
        self.transform = transform
        self.file_paths = []
        self.labels = []
        self.base_path = base_path  # Store base_path as instance variable
        self.target_length = 16000 * 5  # 5 seconds at 16kHz

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Dataset path does not exist: {base_path}")
        if not os.path.exists(protocol_path):
            raise FileNotFoundError(f"Protocol file does not exist: {protocol_path}")

        label_dict = {}
        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=2)
                if len(parts) < 2:
                    continue
                filename = parts[0].strip()
                label_type = parts[1].strip()
                label_dict[filename] = 1 if label_type == 'genuine' else 0

        wav_files = sorted([f for f in os.listdir(base_path) if f.endswith('.wav')])
        for wav_file in wav_files:
            if wav_file in label_dict:
                self.file_paths.append(os.path.join(base_path, wav_file))
                self.labels.append(label_dict[wav_file])

        # Add fixed STFT parameters as class variables
        self.n_fft = 2048
        self.win_length = 2048
        self.hop_length = 441  # Fixed hop length to get desired time steps
        self.fixed_length = 16000 * 5  # 5 seconds at 16kHz
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=224,
            f_min=0,
            f_max=8000,
            power=2
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db = 80)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            # Load audio using torchaudio
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Resample if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Ensure 5 seconds length (16000 * 5 samples)
            target_length = 16000 * 5
            if waveform.shape[1] < target_length:
                waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
            else:
                waveform = waveform[:, :target_length]

            # Generate mel spectrogram
            mel_spec = self.mel_transform(waveform)
            mel_spec_db = self.amplitude_to_db(mel_spec)
            
            # Normalize to [0, 1]
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            # Convert to 3 channels
            mel_spec_db = mel_spec_db.repeat(3, 1, 1)
            
            # Resize to 224x224 using torch interpolate
            mel_spec_db = torch.nn.functional.interpolate(
                mel_spec_db.unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # Apply additional transforms if any
            if self.transform:
                mel_spec_db = self.transform(mel_spec_db)
            
            return mel_spec_db, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.long)

# Add EarlyStopping class before train_model function
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased to {val_loss:.4f}. Resetting early stopping counter.')
        else:
            self.counter += 1
            if self.verbose:
                print(f'Validation loss did not decrease. Early stopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

# Add evaluate_model function before train_model function
def evaluate_model(model, data_loader, criterion, device):
    """Evaluate the model and compute metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1

def train_model(model, train_loader, dev_loader, criterion, optimizer, device, epochs=10, patience=5):
    """Training function without problematic autocast"""
    torch.cuda.empty_cache()
    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        valid_batches = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
            try:
                if inputs.size(0) == 0:
                    continue
                
                # Ensure inputs are float32 and on device
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device)
                
                # Regular forward pass without autocast
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Regular backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                valid_batches += 1
                
                if batch_idx % 10 == 0:
                    acc = 100. * correct / total if total > 0 else 0
                    print(f'Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {acc:.2f}%')
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        # Epoch statistics
        epoch_loss = running_loss / valid_batches if valid_batches > 0 else float('inf')
        epoch_acc = 100. * correct / total if total > 0 else 0
        print(f'\nEpoch {epoch+1}: Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%')
        
        # Validation phase
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, dev_loader, criterion, device)
        print(f'Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            model.load_state_dict(early_stopping.best_model_wts)
            break
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved new best model")
    
    return model

# Add weighted loss for class imbalance
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=[1.0, 2.0]):
        super().__init__()
        self.weights = torch.FloatTensor(weights)
        
    def forward(self, outputs, targets):
        self.weights = self.weights.to(outputs.device)
        return F.cross_entropy(outputs, targets, weight=self.weights)  # 'F' is now defined

# Main Execution
if __name__ == "__main__":
    # Base directory paths
    base_path = "D:/Digital Audio Forensics/Dataset/DS_10283_3055"

    train_audio_path = os.path.join(base_path, "ASVspoof2017_V2_train", "ASVspoof2017_V2_train")
    train_protocol_path = os.path.join(base_path, "protocol_V2", "ASVspoof2017_V2_train.trn.txt")
    dev_audio_path = os.path.join(base_path, "ASVspoof2017_V2_dev", "ASVspoof2017_V2_dev")
    dev_protocol_path = os.path.join(base_path, "protocol_V2", "ASVspoof2017_V2_dev.trl.txt")
    eval_audio_path = os.path.join(base_path, "ASVspoof2017_V2_eval", "ASVspoof2017_V2_eval")
    eval_protocol_path = os.path.join(base_path, "protocol_V2", "ASVspoof2017_V2_eval.trl.txt")

    # Verify paths exist
    print("Checking paths:")
    for path in [train_audio_path, train_protocol_path, dev_audio_path, dev_protocol_path, eval_audio_path, eval_protocol_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")
        print(f"Found: {path}")

    data_transforms = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.485, 0.485],
            std=[0.229, 0.229, 0.229]
        )
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = AudioDataset(
        base_path=train_audio_path,
        protocol_path=train_protocol_path,
        transform=data_transforms
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Reduced batch size
        shuffle=True,
        num_workers=0,  # Single worker
        pin_memory=True
    )

    dev_dataset = AudioDataset(
        base_path=dev_audio_path,
        protocol_path=dev_protocol_path,
        transform=data_transforms
    )
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

    eval_dataset = AudioDataset(
        base_path=eval_audio_path,
        protocol_path=eval_protocol_path,
        transform=data_transforms
    )
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    model = res2net101_26w_4s(pretrained=True, num_classes=2)
    model = model.to(device)

    criterion = WeightedCrossEntropyLoss(weights=[1.0, 2.0])  # Adjust weights based on your class distribution
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.01)
    
    # Train the model
    model = train_model(model, train_loader, dev_loader, criterion, optimizer, device)

    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in eval_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)

    print("Predictions complete.")


