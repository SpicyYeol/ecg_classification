from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d.proj3d import transform
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
# Additional libraries for data transformation
import numpy as np
from pyts.image import RecurrencePlot, GramianAngularField, MarkovTransitionField

from ecg_cluster import load_and_preprocess_ecg_data
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import optuna

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OFFSET = None
DEBUG = False
LEARNING_RATE = 0.001
PLOT = False
dtype = 1
CLUSTERING = True

input_dim = 1  # ECG 시퀀스의 단일 차원 입력
d_model = 32  # Transformer 모델 차원
nhead = 4  # Multi-head attention
num_layers = 4  # Transformer 레이어 수
dim_feedforward = 128  # FFN의 차원
seq_length = 300  # ECG 시퀀스 길이
batch_size = 32
num_epochs = 10
# num_classes = 4  # a, b, c, d에 대한 4개 클래스

# option = ['TRAIN', 'VAL', 'HEATMAP']
option = ['TRAIN']


def augment_ecg_signal(signal):
    # Gaussian noise addition
    noise = np.random.normal(0, 0.01, signal.shape)
    augmented_signal = signal + noise
    return augmented_signal

def normalize_signal(signal):
    # Signal normalization to [0,1]
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = (signal - min_val) / (max_val - min_val + 1e-8)
    return normalized_signal

# Define the Dataset class
class ECGDataset(Dataset):
    def __init__(self, data, labels, augment = False):
        """
        data: list of numpy arrays, each of shape (N_i, 300)
        labels: numpy array of shape (Batch,)
        """
        self.data = data  # List of arrays of shape (N_i, 300)
        self.labels = labels  # Shape: (Batch,)
        self.augment = augment
        
        # Initialize the transformers
        self.transformers = {
            'RP': RecurrencePlot(),
            'GASF': GramianAngularField(method='summation'),
            'GADF': GramianAngularField(method='difference'),
            'MTF': MarkovTransitionField(),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signals = self.data[idx]  # Shape: (N_i, 300)
        label = self.labels[idx]

        transformed_signals = []

        if self.augment:
            # Apply augmentation to each signal
            signals = [augment_ecg_signal(signal) for signal in signals]
            # Optionally apply other augmentations
            signals = [normalize_signal(signal) for signal in signals]

        # Apply transformations to each of the N_i signals
        for signal in signals:  # Iterate over N_i signals
            signal_transformed = []
            for key in ['RP', 'GASF', 'GADF','MTF']:
                transformer = self.transformers[key]
                img = transformer.transform(signal.reshape(1, -1))[0]  # Shape: (image_size, image_size)
                signal_transformed.append(img)
            # Stack the three transformed images along the channel dimension
            signal_transformed = np.stack(signal_transformed, axis=0)  # Shape: (3, image_size, image_size)
            transformed_signals.append(signal_transformed)

        # transformed_signals is a list of length N_i, each element of shape (3, image_size, image_size)

        # Convert to torch tensors
        transformed_signals = [torch.tensor(ts, dtype=torch.float32).to(device) for ts in transformed_signals]
        label = torch.tensor(label, dtype=torch.long).to(device)

        return transformed_signals, label

# Custom collate function to handle variable-length sequences
def collate_fn(batch):
    """
    batch: list of tuples (transformed_signals, label)
    """
    labels = []
    all_transformed_signals = []
    lengths = []
    for transformed_signals, label in batch:
        labels.append(label)
        all_transformed_signals.append(transformed_signals)
        lengths.append(len(transformed_signals))

    # Find the maximum length (max_N) in this batch
    max_length = max(lengths)

    # Pad sequences and create attention masks
    padded_sequences = []
    attention_masks = []
    for signals in all_transformed_signals:
        N_i = len(signals)
        # Pad signals with zeros to match max_length
        pad_size = max_length - N_i
        if pad_size > 0:
            pad = [torch.zeros_like(signals[0]).to(device) for _ in range(pad_size)]
            signals.extend(pad)
        padded_sequences.append(torch.stack(signals))  # Shape: (max_length, 3, H, W)
        # Attention mask: 1 for valid positions, 0 for padding
        attention_mask = [1] * N_i + [0] * pad_size
        attention_masks.append(attention_mask)

    # Stack into tensors
    padded_sequences = torch.stack(padded_sequences)  # Shape: (batch_size, max_length, 3, H, W)
    attention_masks = torch.tensor(attention_masks, dtype=torch.bool).to(device)  # Shape: (batch_size, max_length)
    labels = torch.stack(labels)

    return padded_sequences, attention_masks, labels


# Hybrid Feature Extractor combining CNN and Transformer
class HybridFeatureExtractor(nn.Module):
    def __init__(self, output_dim=256, dropout_rate=0.3):
        super(HybridFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc = nn.Linear(64 * 8 * 8, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc(x))
        return x  # Output shape: (batch_size, output_dim)

# Label Smoothing Cross Entropy Loss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_probs = nn.LogSoftmax(dim=-1)(inputs)
        # Convert targets to one-hot
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.smoothing) * targets + self.smoothing / num_classes
        loss = (-targets * log_probs).mean()
        return loss

# ECG Transformer Model
class ECGTransformer(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=4, num_layers=2, dropout_rate=0.1):
        super(ECGTransformer, self).__init__()
        self.feature_extractor = HybridFeatureExtractor(output_dim=d_model, dropout_rate=dropout_rate)
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dropout=dropout_rate, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(self.d_model, num_classes)

    def forward(self, x, attention_mask):
        """
        x: tensor of shape (batch_size, max_length, 3, H, W)
        attention_mask: tensor of shape (batch_size, max_length)
        """
        batch_size, max_length, C, H, W = x.size()
        x = x.view(batch_size * max_length, C, H, W)
        x = self.feature_extractor(x)  # Shape: (batch_size * max_length, d_model)
        x = x.view(batch_size, max_length, self.d_model)

        # Create src_key_padding_mask for Transformer
        # In PyTorch's Transformer, src_key_padding_mask is of shape (batch_size, seq_len)
        src_key_padding_mask = ~attention_mask  # Shape: (batch_size, max_length)

        # Pass through Transformer Encoder
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Aggregate the outputs
        attention_mask = attention_mask.unsqueeze(-1)  # Shape: (batch_size, max_length, 1)
        masked_output = output * attention_mask  # Zero out the padding positions
        sum_output = masked_output.sum(dim=1)  # Sum over sequence length
        valid_lengths = attention_mask.sum(dim=1)  # Sum of valid positions
        avg_output = sum_output / valid_lengths  # Average over valid positions

        logits = self.fc(avg_output)
        return logits


# Example of preparing data and labels
# Replace this with your actual data loading mechanism
Batch = 16  # Batch size
# Generate variable-length data

ecg_data_A = load_and_preprocess_ecg_data(OFFSET, [2], dtype, DEBUG, CLUSTERING, PLOT)[:1500]
# ecg_data_B = load_and_preprocess_ecg_data(OFFSET, [4], dtype, DEBUG, CLUSTERING, PLOT)

# 샘플 ECG 데이터 (랜덤 데이터로 대체)
# ecg_data_A = torch.randn(batch_size, seq_length, input_dim)  # A Dataset (a, b, c, d)
# ecg_data_B = torch.randn(batch_size, seq_length, input_dim)  # B Dataset (e, f, g)

# 샘플 레이블 (랜덤 생성된 예시 레이블)
label_to_idx_A = {label: idx for idx, label in enumerate(set(d['label'] for d in ecg_data_A))}
data = [d['data'] for d in ecg_data_A]
labels = np.array([label_to_idx_A[d['label']] for d in ecg_data_A])

# labels_B = torch.randint(0, 3, (batch_size,))  # e, f, g 레이블

# data = []
# labels = []
# for _ in range(Batch):
#     N_i = np.random.randint(5, 15)  # Variable N between 5 and 15
#     signals = np.random.rand(N_i, 300)
#     data.append(signals)
#     labels.append(np.random.randint(0, 2))
# labels = np.array(labels)

from sklearn.model_selection import train_test_split
train_data, val_data, train_labels, val_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42)

from sklearn.utils.class_weight import compute_class_weight
# 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_tensor  = torch.tensor(class_weights, dtype=torch.float32)
# class_weights = class_weights.to(device)

# Optuna objective function
def objective(trial):
    # Hyperparameter suggestions
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    nhead = trial.suggest_categorical('nhead', [4, 8])
    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)

    # Create datasets and dataloaders
    train_dataset = ECGDataset(train_data, train_labels, augment=True)
    val_dataset = ECGDataset(val_data, val_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Initialize the model
    num_classes = 4
    model = ECGTransformer(num_classes=num_classes, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout_rate=dropout_rate)
    model = model.to(device)
    class_weights_tensor_device = class_weights_tensor.to(device)

    # Define the loss function and optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler with warmup and cosine annealing
    num_epochs = 5
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0) if step < warmup_steps else 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    )

    # Training and validation loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        for inputs, attention_mask, labels in train_loader:
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)
            total_samples += labels.size(0)

            # Memory management
            del inputs, attention_mask, labels, outputs, loss
            torch.cuda.empty_cache()

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        # Validation phase
        model.eval()
        val_running_corrects = 0
        val_total_samples = 0
        with torch.no_grad():
            for inputs, attention_mask, labels in val_loader:
                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outputs = model(inputs, attention_mask)

                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels)
                val_total_samples += labels.size(0)

                # Memory management
                del inputs, attention_mask, labels, outputs
                torch.cuda.empty_cache()

        val_epoch_acc = val_running_corrects.double() / val_total_samples

        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {epoch_acc:.4f}, Val Acc: {val_epoch_acc:.4f}")

    # Return the final validation accuracy
    return val_epoch_acc.item()

if __name__ == '__main__':
    freeze_support()
    # Optuna study creation and optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    print("Best hyperparameters:", study.best_params)