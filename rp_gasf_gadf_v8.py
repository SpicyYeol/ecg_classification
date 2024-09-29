import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from linformer import Linformer
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from ecg_cluster import load_and_preprocess_ecg_data
from rp_gasf_gadf_v7 import align_ecg_signals, normalize_signals, transform_to_images_grouped_with_labels, \
    split_dataset_for_deep_learning

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

# Custom Dataset for loading the video data
class VideoDataset(Dataset):
    def __init__(self, grouped_segment, grouped_labels, resize_shape=(100, 100)):
        self.grouped_segments = grouped_segment
        self.grouped_labels = grouped_labels
        self.resize_shape = resize_shape
        self.transform = transforms.Resize(self.resize_shape)

    def __len__(self):
        return len(self.grouped_labels)

    def __getitem__(self, idx):
        imgs = torch.tensor(self.grouped_segments[idx], dtype=torch.float32)
        resized_sequence = torch.stack([self.transform(frame) for frame in imgs])
        return torch.tensor(resized_sequence, dtype=torch.float32), torch.tensor(self.grouped_labels[idx],
                                                                                 dtype=torch.long)


def pad_or_truncate_sequence(sequence, target_length=100):
    sequence_length = sequence.size(0)

    if sequence_length > target_length:
        return sequence[:target_length]  # Truncate the sequence if it's too long
    else:
        padding = target_length - sequence_length
        return torch.cat([sequence, torch.zeros(padding, sequence.size(1), sequence.size(2),
                                                sequence.size(3))])  # Pad if it's too short


def collate_fn(batch):
    data, labels = zip(*batch)

    # Apply truncation or padding to each sample to ensure fixed length
    data = [pad_or_truncate_sequence(d, target_length=100) for d in data]

    # Stack data and create a batch
    data_padded = torch.stack(data, dim=0)
    labels = torch.tensor(labels)

    return data_padded, labels


# Define the dilated CNN with residual connections
class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# Positional Encoding for Transformers
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


# Relational Encoding to capture pairwise relations
class RelationalEncoding(nn.Module):
    def __init__(self, d_model):
        super(RelationalEncoding, self).__init__()
        self.fc_rel = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        batch_size, seq_length, d_model = x.size()
        relations = []
        for i in range(seq_length):
            for j in range(i + 1, seq_length):
                pair = torch.cat([x[:, i, :], x[:, j, :]], dim=-1)  # Concatenate features from i and j
                relations.append(self.fc_rel(pair))
        relations = torch.stack(relations, dim=1)  # Shape: (batch_size, seq_length * (seq_length-1) / 2, d_model)
        return relations


class TransformerEncoderLayerWithAttention(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderLayerWithAttention, self).__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


# Define the model with Dilated CNN and Linformer for the Transformer
class VideoProcessingModel(nn.Module):
    def __init__(self, num_classes=10, d_model=512, nhead=8, num_encoder_layers=2, seq_len=100):
        super(VideoProcessingModel, self).__init__()

        # Dilated CNN layers for spatial feature extraction
        self.layer1 = DilatedResidualBlock(in_channels=3, out_channels=64, dilation=1)
        self.layer2 = DilatedResidualBlock(64, 128, dilation=2)
        self.layer3 = DilatedResidualBlock(128, 256, dilation=4)
        self.layer4 = DilatedResidualBlock(256, 512, dilation=8)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Positional Encoding for the Transformer
        self.pos_encoder = PositionalEncoding(d_model)

        # Linformer-based Transformer Encoder
        self.transformer_encoder = Linformer(
            dim=d_model,        # Model dimension
            seq_len=seq_len,    # Sequence length (e.g., 100)
            depth=num_encoder_layers,  # Number of layers
            heads=nhead,        # Number of attention heads
            k=128               # Low-rank approximation
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()

        # Apply Dilated CNN to each frame
        x = x.view(batch_size * seq_length, channels, height, width)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)
        x = x.view(batch_size, seq_length, -1)  # Flatten CNN output for each frame

        # Apply positional encoding
        x = self.pos_encoder(x)

        # Apply Transformer Encoder directly
        x = self.transformer_encoder(x)

        # Average the transformer output over time and classify
        output = self.fc(x.mean(dim=1))

        return output


# Train function with DataLoader and validation process
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, accumulation_steps=4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()  # For mixed precision training

    for epoch in range(epochs):
        # Training process
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()  # Clear gradients

        for i, (batch_data, batch_labels) in enumerate(train_loader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)


            with autocast():  # Mixed precision
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss = loss / accumulation_steps  # Scale loss for gradient accumulation

            scaler.scale(loss).backward()  # Backward pass with scaled gradients

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)  # Update weights
                scaler.update()  # Update the scaler
                optimizer.zero_grad()  # Clear gradients for next accumulation step

            total_train_loss += loss.item()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}')

        # Validation process
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                with autocast():  # Mixed precision
                    val_outputs = model(batch_data)
                    val_loss = criterion(val_outputs, batch_labels)

                total_val_loss += val_loss.item()
                _, predicted = torch.max(val_outputs, 1)
                correct_predictions += (predicted == batch_labels).sum().item()
                total_predictions += batch_labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions * 100
        print(
            f'Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {val_accuracy:.2f}%')


# Main function to run the model
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 4
    seq_length = 10  # Variable length can be handled in real applications
    num_classes = 10
    epochs = 100
    lr = 0.001

    OFFSET = None
    DEBUG = False
    dtype = 1
    CLUSTERING = True
    PLOT = False

    try:
        ecg_data_A = load_and_preprocess_ecg_data(OFFSET, [2], dtype, DEBUG, CLUSTERING, PLOT)[:100]
    except Exception as e:
        logging.error(f"Data loading error: {e}")

    label_to_idx_A = {label: idx for idx, label in enumerate(set(d['label'] for d in ecg_data_A))}
    data = [d['data'] for d in ecg_data_A]
    labels = np.array([label_to_idx_A[d['label']] for d in ecg_data_A])

    num_classes = len(label_to_idx_A)
    print(f"Number of classes: {num_classes}")

    # Data filtering and alignment
    finite_mask = np.isfinite(labels)
    data = [d for d, f in zip(data, finite_mask) if f]
    labels = labels[finite_mask]

    aligned_signals, aligned_labels = align_ecg_signals(data, labels, sample_rate=100, window_size=100)

    normalized_signals = normalize_signals(aligned_signals)

    image_dataset, source_mapping, labels_per_segment = transform_to_images_grouped_with_labels(
        grouped_segments=normalized_signals,
        grouped_labels=aligned_labels
    )

    train_grouped_images, val_grouped_images, train_grouped_segments, val_grouped_segments, train_grouped_labels, val_grouped_labels = split_dataset_for_deep_learning(
        image_dataset=image_dataset,
        source_mapping=source_mapping,
        labels_per_segment=labels_per_segment,
        grouped_segments=normalized_signals,
        grouped_labels=aligned_labels,
        train_size=0.8,
        random_state=42,
        return_grouped=True
    )

    # Initialize model
    model = VideoProcessingModel(num_classes=num_classes).to(device)

    # Create DataLoader for train and validation
    train_dataset = VideoDataset(train_grouped_images, train_grouped_labels)
    val_dataset = VideoDataset(val_grouped_images, val_grouped_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    # Train the model with validation after each epoch
    train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)
