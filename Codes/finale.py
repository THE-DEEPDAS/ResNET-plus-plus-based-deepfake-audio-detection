import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import noisereduce as nr
from torchvision import models, transforms
from PIL import Image
import os
import cv2
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import copy  # Added for deepcopy in early stopping
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch import amp  # Update import for newer PyTorch versions
from einops import rearrange, reduce
import torchaudio
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import tensorflow as tf
import kapre
from kapre.composed import get_melspectrogram_layer
from sklearn.metrics import confusion_matrix
import librosa.effects
import librosa.display
import scipy.signal as signal
from scipy.interpolate import interp1d

# Preprocessing Functions
def load_audio(file_path, sr=16000):
    """Load any audio format and resample."""
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        return audio, sample_rate
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def normalize_audio(audio):
    """Normalize audio to have zero mean and unit variance."""
    return (audio - np.mean(audio)) / np.std(audio)

def remove_gaussian_noise(audio, sr):
    """Apply spectral gating to remove Gaussian noise."""
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    return reduced_noise

def apply_histogram_equalization(mel_spectrogram):
    """Apply histogram equalization to mel spectrogram."""
    mel_spectrogram_norm = cv2.normalize(mel_spectrogram, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.equalizeHist(mel_spectrogram_norm)

def extract_mel_spectrogram(audio, sr, n_mels=128, n_fft=2048, hop_length=512):
    """Extract mel-spectrogram with histogram equalization."""
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_eq = apply_histogram_equalization(S_dB)
    return S_eq

# Enhanced Preprocessing Functions
def apply_pre_emphasis(audio, coef=0.97):
    """Apply pre-emphasis filter to audio"""
    return np.append(audio[0], audio[1:] - coef * audio[:-1])

def apply_mvn(features):
    """Apply Mean Variance Normalization"""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-12
    return (features - mean) / std

def apply_cmvn(features):
    """Apply Cepstral Mean Variance Normalization"""
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True) + 1e-12
    return (features - mean) / std

def apply_delta_features(features, N=2):
    """Compute delta and delta-delta features"""
    delta = librosa.feature.delta(features, order=1, width=N*2+1)
    delta2 = librosa.feature.delta(features, order=2, width=N*2+1)
    return np.concatenate([features, delta, delta2], axis=0)

def apply_energy_vad(audio, sr, frame_length=2048, hop_length=512, threshold=0.5):
    """Voice Activity Detection based on energy"""
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)
    energy_mean = np.mean(energy)
    return audio[energy.squeeze() > (energy_mean * threshold)]

def enhanced_mel_spectrogram(audio, sr, n_mels=128):
    """Enhanced mel spectrogram using Keras/Kapre"""
    # Convert to tensor
    audio_tensor = tf.convert_to_tensor(audio[np.newaxis, :], dtype=tf.float32)
    
    # Create mel spectrogram layer
    mel_layer = get_melspectrogram_layer(
        input_shape=(None,),
        n_mels=n_mels,
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        mel_f_min=0.0,
        mel_f_max=sr/2,
        return_decibel=True,
        input_data_format='channels_last',
        output_data_format='channels_last'
    )
    
    # Get mel spectrogram
    mel_spec = mel_layer(audio_tensor)
    return mel_spec.numpy()[0]

# Dataset Class
class AudioDataset(Dataset):
    def _init_(self, base_path, protocol_path, transform=None, sr=16000):
        self.transform = transform
        self.file_paths = []
        self.labels = []
        self.sr = sr
        self.base_path = base_path  # Store base_path as instance variable
        
        print(f"Loading dataset from: {base_path}")
        print(f"Using protocol file: {protocol_path}")
        
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Dataset path does not exist: {base_path}")
        if not os.path.exists(protocol_path):
            raise FileNotFoundError(f"Protocol file does not exist: {protocol_path}")
            
        # Load protocol file first
        label_dict = {}
        with open(protocol_path, 'r') as f:
            for line in f:
                # Split on whitespace but keep filename as first element
                parts = line.strip().split(maxsplit=2)
                if len(parts) < 2:
                    print(f"Warning: Malformed line in protocol file: {line}")
                    continue
                    
                filename = parts[0].strip()  # Get filename without leading/trailing whitespace
                label_type = parts[1].strip()  # Get label type (genuine or spoof)
                label_dict[filename] = 1 if label_type == 'genuine' else 0
                
        print(f"Loaded {len(label_dict)} entries from protocol file")
        print("First few protocol entries:")
        for i, (k, v) in enumerate(label_dict.items()):
            if i < 5:
                print(f"{k}: {'genuine' if v==1 else 'spoof'}")
        
        # Scan directory for audio files
        wav_files = sorted([f for f in os.listdir(base_path) if f.endswith('.wav')])
        print(f"\nFound {len(wav_files)} WAV files in directory")
        print("First few WAV files:")
        for f in wav_files[:5]:
            print(f)
            
        # Match files with protocol entries
        for wav_file in wav_files:
            if wav_file in label_dict:
                self.file_paths.append(os.path.join(base_path, wav_file))
                self.labels.append(label_dict[wav_file])
            else:
                print(f"Warning: No protocol entry for file {wav_file}")
        
        # Print final statistics
        genuine = self.labels.count(1)
        spoofed = self.labels.count(0)
        print("\nDataset Statistics:")
        print(f"Number of genuine samples: {genuine}")
        print(f"Number of spoofed samples: {spoofed}")
        print(f"Total files matched: {len(self.file_paths)}")
        
        if len(self.file_paths) == 0:
            raise RuntimeError("No files were matched between directory and protocol!")

    def _len_(self):
        return len(self.file_paths)

    def _getitem_(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            # Load and process audio
            audio, sr = load_audio(file_path, self.sr)
            if audio is None or sr is None:
                # Create valid dummy tensor instead of zeros
                dummy_input = torch.randn(3, 224, 224)  # Random noise is better than zeros
                print(f"Warning: Creating dummy tensor for {file_path}")
                return dummy_input, torch.tensor(label, dtype=torch.long)

            # Process audio
            audio = normalize_audio(audio)
            audio = remove_gaussian_noise(audio, sr)
            mel_spectrogram = extract_mel_spectrogram(audio, sr)
            
            # Ensure mel_spectrogram is not None and has correct shape
            if mel_spectrogram is None or mel_spectrogram.size == 0:
                dummy_input = torch.randn(3, 224, 224)
                print(f"Warning: Invalid mel spectrogram for {file_path}")
                return dummy_input, torch.tensor(label, dtype=torch.long)
            
            # Apply augmentation
            if self.transform and 'train' in str(self.base_path).lower():
                mel_spectrogram = spec_augment(mel_spectrogram)
            
            # Convert to image format
            mel_spectrogram = np.stack([mel_spectrogram] * 3, axis=-1)
            mel_spectrogram = Image.fromarray(np.uint8(mel_spectrogram))

            if self.transform:
                mel_spectrogram = self.transform(mel_spectrogram)
            
            # Ensure tensor is valid
            if not isinstance(mel_spectrogram, torch.Tensor):
                dummy_input = torch.randn(3, 224, 224)
                print(f"Warning: Transform failed for {file_path}")
                return dummy_input, torch.tensor(label, dtype=torch.long)
            
            return mel_spectrogram, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            dummy_input = torch.randn(3, 224, 224)
            return dummy_input, torch.tensor(label, dtype=torch.long)

# Add SEBlock class
class SEBlock(nn.Module):
    def _init_(self, channel, reduction=16):
        super(SEBlock, self)._init_()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Add new attention and residual blocks
class ChannelAttention(nn.Module):
    def _init_(self, in_channels, reduction=16):
        super(ChannelAttention, self)._init_()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def _init_(self, kernel_size=7):
        super(SpatialAttention, self)._init_()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        return self.sigmoid(attn)

class CBAM(nn.Module):
    def _init_(self, channels, reduction=16):
        super(CBAM, self)._init_()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.spatial_attention(spatial)
        
        return x * spatial

# Define ArcMarginProduct
class ArcMarginProduct(nn.Module):
    """
    Implements margin-based softmax: ArcFace (https://arxiv.org/abs/1801.07698)
    """
    def _init_(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self)._init_()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x, labels):
        # Normalize features & weights
        x_norm = F.normalize(x)
        w_norm = F.normalize(self.weight)
        # Cosine of angle between x and w
        cos_theta = torch.mm(x_norm, w_norm.t())
        # Add margin
        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
        target_logits = torch.cos(theta + self.m)
        # Build output
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = cos_theta * (1 - one_hot) + target_logits * one_hot
        # Scale
        output *= self.s
        return output

# Define ResidualBlock
class ResidualBlock(nn.Module):
    def _init_(self, in_channels, out_channels):
        super(ResidualBlock, self)._init_()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Optional downsample if in/out channels differ
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# Define TransformerBlock
class TransformerBlock(nn.Module):
    def _init_(self, dim, num_heads=8, mlp_ratio=4., dropout=0.):
        super()._init_()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(*[self.norm1(x)]*3)[0]
        x = x + self.mlp(self.norm2(x))
        return x

# Define SETransformerBlock
class SETransformerBlock(nn.Module):
    def _init_(self, channels, reduction=16):
        super()._init_()
        self.se = SEBlock(channels, reduction)
        self.transformer = TransformerBlock(channels)
        
    def forward(self, x):
        b, c, h, w = x.shape
        # SE attention
        x = self.se(x)
        # Transformer attention
        x_trans = rearrange(x, 'b c h w -> (h w) b c')
        x_trans = self.transformer(x_trans)
        x_trans = rearrange(x_trans, '(h w) b c -> b c h w', h=h, w=w)
        return x + x_trans

# Define MultiScaleFeatureFusion
class MultiScaleFeatureFusion(nn.Module):
    def _init_(self, channels):
        super()._init_()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=1, dilation=1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(channels//4),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=4, dilation=4),
            nn.BatchNorm2d(channels//4),
            nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return self.fusion(torch.cat([x1, x2, x3, x4], dim=1))

# Define ResNet++ Model (Enhanced ResNet)
class ResNetPlusPlus(nn.Module):
    def _init_(self, num_classes=2):
        super()._init_()
        
        # Base ResNet
        self.base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Modify first conv to handle our input
        self.base.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the original FC layer
        self.base = nn.Sequential(*list(self.base.children())[:-2])
        
        # Add custom layers
        self.attention = CBAM(2048)  # Add attention after final conv block
        
        # Global pooling and FC layers
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        
        # Base feature extraction
        x = self.base(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = self.fc(x)
        
        return x

# Add Metrics Calculation Functions
def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, and F1-score."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

def calculate_detailed_metrics(y_true, y_pred):
    """Calculate detailed metrics including FPR, FNR, TPR, TNR"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Additional metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return accuracy, precision, recall, f1, fpr, fnr, tpr, tnr

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
    accuracy, precision, recall, f1, fpr, fnr, tpr, tnr = calculate_detailed_metrics(all_labels, all_preds)
    
    # Print metrics directly to terminal
    print(f'Evaluation Metrics:')
    print(f'Loss: {avg_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'FPR: {fpr:.4f}')
    print(f'FNR: {fnr:.4f}')
    print(f'TPR: {tpr:.4f}')
    print(f'TNR: {tnr:.4f}')
    
    return avg_loss, accuracy, precision, recall, f1, fpr, fnr, tpr, tnr

# Add EarlyStopping class
class EarlyStopping:
    def _init_(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None

    def _call_(self, val_loss, model):
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

# Training Pipeline
def train_model(model, train_loader, dev_loader, criterion, optimizer, device, epochs=10, patience=5):
    torch.cuda.empty_cache()
    
    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f'\nEpoch {epoch+1}/{epochs}')
        print('-' * 60)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
            try:
                if inputs.size(0) == 0:
                    continue
                    
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 10 == 0:
                    acc = 100. * correct / total
                    print(f'Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {acc:.2f}%')
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Training Accuracy: {epoch_acc:.2f}%')
        
        # Validation phase
        print('\nValidation Phase:')
        val_loss, val_acc, val_prec, val_rec, val_f1, val_fpr, val_fnr, val_tpr, val_tnr = evaluate_model(model, dev_loader, criterion, device)
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            model.load_state_dict(early_stopping.best_model_wts)
            break
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved new best model!")
        
        print('-' * 60)

    return model

# 1) Define FocalLoss
class FocalLoss(nn.Module):
    def _init_(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self)._init_()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 2) SpecAugment transformations (time & frequency masking)
def spec_augment(mel_spectrogram, time_mask_param=20, freq_mask_param=10):
    # Random frequency masking
    freq_mask = random.randint(0, freq_mask_param)
    freq_start = random.randint(0, max(0, mel_spectrogram.shape[0] - freq_mask))
    mel_spectrogram[freq_start:freq_start + freq_mask, :] = 0

    # Random time masking
    time_mask = random.randint(0, time_mask_param)
    time_start = random.randint(0, max(0, mel_spectrogram.shape[1] - time_mask))
    mel_spectrogram[:, time_start:time_start + time_mask] = 0

    return mel_spectrogram

class RawWaveformModule(nn.Module):
    """Process raw waveform using 1D convolutions (from ASVspoof 2021 winning solution)"""
    def _init_(self):
        super()._init_()
        self.conv1 = nn.Conv1d(1, 64, 1024, stride=16)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, stride=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, stride=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, 3, stride=2)
        self.bn4 = nn.BatchNorm1d(512)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x

class LightWeightMFA(nn.Module):
    """Light-weight learnable multi-feature aggregation module"""
    def _init_(self, in_channels, out_channels):
        super()._init_()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(out_channels * 4)
        self.fc = nn.Linear(out_channels * 4, out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.bn(x)
        x = F.relu(x)
        x = torch.mean(x, dim=[2, 3])
        x = self.fc(x)
        return x

class SelfSupervisedContrastiveLearning(nn.Module):
    """Self-supervised contrastive learning module"""
    def _init_(self, feature_dim, projection_dim):
        super()._init_()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()

# Example Usage
if __name__ == "__main__":
    # Base directory paths
    base_path = "C:\Volume D\03 PROGRAMMING\28 Deep\Dataset\DS_10283_3055"
    
    train_audio_path = os.path.join(base_path, "ASVspoof2017_V2_train", "ASVspoof2017_V2_train")
    train_protocol_path = os.path.join(base_path, "protocol_V2", "ASVspoof2017_V2_train.trn.txt")
    dev_audio_path = os.path.join(base_path, "ASVspoof2017_V2_dev", "ASVspoof2017_V2_dev")
    dev_protocol_path = os.path.join(base_path, "protocol_V2", "ASVspoof2017_V2_dev.trl.txt")
    eval_audio_path = os.path.join(base_path, "ASVspoof2017_V2_eval", "ASVspoof2017_V2_eval")
    eval_protocol_path = os.path.join(base_path, "protocol_V2", "ASVspoof2017_V2_eval.trl.txt")

    # Verify paths exist
    print("Checking paths:")
    for path in [train_audio_path, train_protocol_path, 
                 dev_audio_path, dev_protocol_path,
                 eval_audio_path, eval_protocol_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")
        print(f"Found: {path}")
    
    # Look for sample files to verify correct paths
    train_sample = os.path.join(train_audio_path, "T_1000001.wav")
    dev_sample = os.path.join(dev_audio_path, "D_1000001.wav")
    eval_sample = os.path.join(eval_audio_path, "E_1000002.wav")

    for sample in [train_sample, dev_sample, eval_sample]:
        if os.path.exists(sample):
            print(f"Found sample file: {sample}")
        else:
            print(f"Warning: Sample file not found: {sample}")

    # Data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Determine device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Updated to reflect device change
    
    # Prepare dataset and dataloader with corrected paths
    train_dataset = AudioDataset(
        base_path=train_audio_path,
        protocol_path=train_protocol_path,
        transform=data_transforms
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4,  # Adjust based on your CPU cores
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

    # Add batch size validation
    def validate_batch_size(loader, device):
        try:
            inputs, labels = next(iter(loader))
            inputs = inputs.to(device)
            labels = labels.to(device)
            return True
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Warning: Batch size too large for GPU memory")
                return False
        return True

    # Initialize model with simpler architecture
    model = ResNetPlusPlus(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Train the model with validation
    model = train_model(model, train_loader, dev_loader, criterion, optimizer, device)