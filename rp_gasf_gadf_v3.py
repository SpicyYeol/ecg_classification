# 추가 라이브러리
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyts.image import MarkovTransitionField
from pyts.image import RecurrencePlot, GramianAngularField
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
# 혼합 정밀도 학습을 위한 라이브러리
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from ecg_cluster import load_and_preprocess_ecg_data

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OFFSET = None
DEBUG = False
LEARNING_RATE = 0.001
PLOT = False
dtype = 1
CLUSTERING = True

# Hyperparameters는 main() 내에서 정의
# input_dim = 1  # 사용하지 않음
# d_model = 32  # 사용하지 않음
# nhead = 4     # 사용하지 않음
# num_layers = 4  # 사용하지 않음
# dim_feedforward = 128  # 사용하지 않음
# seq_length = 300  # 사용하지 않음
# batch_size = 32    # main()에서 재정의
# num_epochs = 10    # main()에서 재정의
# num_classes = 4    # main()에서 동적으로 정의

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
    def __init__(self, data, labels, augment=False):
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
            for key in ['RP', 'GASF', 'GADF', 'MTF']:
                transformer = self.transformers[key]
                img = transformer.transform(signal.reshape(1, -1))[0]  # Shape: (image_size, image_size)
                signal_transformed.append(img)
            # Stack the four transformed images along the channel dimension
            signal_transformed = np.stack(signal_transformed, axis=0)  # Shape: (4, image_size, image_size)
            signal_transformed = (signal_transformed - np.mean(signal_transformed)) / (
                        np.std(signal_transformed) + 1e-8)
            transformed_signals.append(signal_transformed)

        # transformed_signals is a list of length N_i, each element of shape (4, image_size, image_size)

        # Convert to torch tensors
        transformed_signals = [torch.tensor(ts, dtype=torch.float32) for ts in transformed_signals]
        label = torch.tensor(label, dtype=torch.long)

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
            pad = [torch.zeros_like(signals[0]) for _ in range(pad_size)]
            signals.extend(pad)
        padded_sequences.append(torch.stack(signals))  # Shape: (max_length, 4, H, W)
        # Attention mask: 1 for valid positions, 0 for padding
        attention_mask = [1] * N_i + [0] * pad_size
        attention_masks.append(attention_mask)

    # Stack into tensors
    padded_sequences = torch.stack(padded_sequences)  # Shape: (batch_size, max_length, 4, H, W)
    attention_masks = torch.tensor(attention_masks, dtype=torch.bool)  # Shape: (batch_size, max_length)
    labels = torch.stack(labels)  # Shape: (batch_size,)

    # Move to device
    padded_sequences = padded_sequences.to(device)
    attention_masks = attention_masks.to(device)
    labels = labels.to(device)

    return padded_sequences, attention_masks, labels


# ============================================
# Advanced Attention Mechanism: Relative Positional Encoding
# ============================================

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Initialize relative positional embeddings with small values
        self.rel_pos_table = nn.Parameter(torch.randn(max_len * 2 - 1, d_model) * 0.1)  # 작은 표준편차로 초기화


    def forward(self, length):
        """
        Generate relative positional embeddings for a given sequence length.

        Args:
            length (int): Length of the sequence.

        Returns:
            Tensor: Relative positional embeddings of shape (length, length, d_model)
        """
        range_vec = torch.arange(length, device=self.rel_pos_table.device)
        distance_mat = range_vec.unsqueeze(1) - range_vec.unsqueeze(0)  # Shape: (length, length)
        distance_mat_clipped = torch.clamp(distance_mat + self.max_len - 1, 0, 2 * self.max_len - 2)
        rel_pos_embeddings = self.rel_pos_table[distance_mat_clipped]  # Shape: (length, length, d_model)
        return rel_pos_embeddings


# ============================================
# Transfer Learning: Feature Extractor with Pre-trained ResNet34
# ============================================

class FeatureExtractor(nn.Module):
    def __init__(self, output_dim=256, dropout_rate=0.3):
        super(FeatureExtractor, self).__init__()
        # Load pre-trained ResNet34
        self.backbone = models.resnet34(pretrained=True)
        # Modify the first convolution layer to accept 4 channels
        self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize the 4th channel's weights as the mean of the first three channels
        with torch.no_grad():
            self.backbone.conv1.weight[:, 3, :, :] = self.backbone.conv1.weight[:, :3, :, :].mean(dim=1)
        # Remove the last two layers (avgpool and fc)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, output_dim)

        # Freeze backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size * max_length, 4, H, W)

        Returns:
            Tensor: Feature embeddings of shape (batch_size * max_length, output_dim)
        """
        x = self.backbone(x)  # Shape: (batch_size * max_length, 512, H', W')
        x = self.pool(x)      # Shape: (batch_size * max_length, 512, 1, 1)
        x = x.view(x.size(0), -1)  # Shape: (batch_size * max_length, 512)
        x = self.dropout(x)
        x = self.fc(x)         # Shape: (batch_size * max_length, output_dim)
        return x


# ============================================
# Transformer Model with Advanced Attention
# ============================================
class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, max_relative_position=128, batch_first=False):
        super(CustomMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_relative_position = max_relative_position
        self.batch_first = batch_first

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Relative positional encoding
        self.relative_pos_embedding = RelativePositionalEncoding(self.head_dim, max_len=max_relative_position)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, is_causal=None):
        """
        Args:
            query, key, value: (seq_length, batch_size, embed_dim) if batch_first=False
            attn_mask: (batch_size, num_heads, seq_length, seq_length)
            key_padding_mask: (batch_size, seq_length)
            is_causal: bool (unused, for compatibility)
        Returns:
            attn_output: (seq_length, batch_size, embed_dim)
        """
        if self.batch_first:
            batch_size, seq_length, embed_dim = query.size()
        else:
            seq_length, batch_size, embed_dim = query.size()

        # Query, Key, Value projections
        q = self.q_proj(query)  # (seq_length, batch_size, embed_dim)
        k = self.k_proj(key)    # (seq_length, batch_size, embed_dim)
        v = self.v_proj(value)  # (seq_length, batch_size, embed_dim)

        # Split into multiple heads
        q = q.view(seq_length, batch_size, self.num_heads, self.head_dim).transpose(1, 2)  # (seq_length, num_heads, batch_size, head_dim)
        k = k.view(seq_length, batch_size, self.num_heads, self.head_dim).transpose(1, 2)  # (seq_length, num_heads, batch_size, head_dim)
        v = v.view(seq_length, batch_size, self.num_heads, self.head_dim).transpose(1, 2)  # (seq_length, num_heads, batch_size, head_dim)

        # Adjust shapes for batch processing
        q = q.transpose(0, 2)  # (batch_size, num_heads, seq_length, head_dim)
        k = k.transpose(0, 2)  # (batch_size, num_heads, seq_length, head_dim)
        v = v.transpose(0, 2)  # (batch_size, num_heads, seq_length, head_dim)

        # Generate relative positional embeddings
        rel_pos = self.relative_pos_embedding(seq_length)  # (seq_length, seq_length, head_dim)
        rel_pos = rel_pos.permute(2, 0, 1)  # (head_dim, seq_length, seq_length)

        # Calculate relative attention
        rel_attn = torch.einsum('bnhd, dkl -> bnhl', q, rel_pos)  # (batch_size, num_heads, seq_length, seq_length)

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(
            self.head_dim)  # (batch_size, num_heads, seq_length, seq_length)
        attn_scores += rel_attn  # Add relative positional encoding

        if attn_mask is not None:
            # Convert mask to boolean
            mask_neg_inf = (attn_mask == float('-inf')).bool()
            mask_inf = (attn_mask == float('inf')).bool()
            attn_scores = attn_scores.masked_fill(mask_neg_inf, float('-inf'))
            attn_scores = attn_scores.masked_fill(mask_inf, float('inf'))

        if key_padding_mask is not None:
            # Ensure key_padding_mask is boolean
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).bool(), float('-inf')
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)
        attn_weights = torch.dropout(attn_weights, p=self.dropout, train=self.training)

        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_length, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)  # (batch_size, seq_length, embed_dim)

        attn_output = self.out_proj(attn_output)  # (batch_size, seq_length, embed_dim)

        # Fix the shape to match the input (seq_length, batch_size, embed_dim)
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)  # (seq_length, batch_size, embed_dim)

        return attn_output



class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)

        # Position-wise Feedforward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
                              is_causal=is_causal)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class ECGTransformer(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=2, dropout_rate=0.3):
        super(ECGTransformer, self).__init__()
        self.feature_extractor = FeatureExtractor(output_dim=d_model, dropout_rate=dropout_rate)
        self.feature_norm = nn.LayerNorm(d_model)  # LayerNorm 추가
        self.d_model = d_model

        # Define transformer encoder
        encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(self.d_model, num_classes)

    def forward(self, x, attention_mask):
        """
        Args:
            x (Tensor): Input tensor, shape (batch_size, max_length, 4, H, W)
            attention_mask (Tensor): Attention mask, shape (batch_size, max_length)
        Returns:
            Tensor: Logits, shape (batch_size, num_classes)
        """
        batch_size, max_length, C, H, W = x.size()
        # Debugging: Print input shape
        # print(f"Input shape before view: {x.shape}")

        x = x.view(batch_size * max_length, C, H, W)  # Combine batch and sequence dimensions
        x = self.feature_extractor(x)  # Shape: (batch_size * max_length, d_model)
        # print(f"Feature Extractor Output: min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}")

        # Apply LayerNorm
        x = self.feature_norm(x)

        # Check for NaN or Inf in feature extractor output
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Feature Extractor Error.")
            raise ValueError("Feature Extractor produced NaN or Inf.")

        x = x.view(batch_size, max_length, self.d_model)  # Shape: (batch_size, max_length, d_model)

        # Transformer expects (seq_length, batch_size, d_model)
        x = x.transpose(0, 1)  # Shape: (max_length, batch_size, d_model)

        # Create key padding mask: True for positions that are to be masked
        src_key_padding_mask = ~attention_mask  # Shape: (batch_size, max_length)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(x,
                                                      src_key_padding_mask=src_key_padding_mask)  # Shape: (max_length, batch_size, d_model)

        # Check for NaN or Inf in transformer output
        if torch.isnan(transformer_output).any() or torch.isinf(transformer_output).any():
            print("Transformer Error.")
            raise ValueError("Transformer produced NaN or Inf.")

        # Transpose back to (batch_size, max_length, d_model)
        transformer_output = transformer_output.transpose(0, 1)  # Shape: (batch_size, max_length, d_model)
        # print(
        #     f"Transformer Output: min={transformer_output.min().item()}, max={transformer_output.max().item()}, mean={transformer_output.mean().item()}")

        # Aggregate using attention mask
        attention_mask = attention_mask.unsqueeze(-1)  # Shape: (batch_size, max_length, 1)
        masked_output = transformer_output * attention_mask  # Invalidate padding tokens
        sum_output = masked_output.sum(dim=1)  # Sum over sequence length
        valid_lengths = attention_mask.sum(dim=1) + 1e-8  # Avoid division by zero
        avg_output = sum_output / valid_lengths  # Average pooling

        logits = self.fc(avg_output)  # Shape: (batch_size, num_classes)
        # print(f"Logits: min={logits.min().item()}, max={logits.max().item()}, mean={logits.mean().item()}")

        # Check for NaN or Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Logits Error.")
            raise ValueError("Logits contain NaN or Inf.")

        return logits


# ============================================
# Training and Validation Functions
# ============================================

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, attention_mask, labels in dataloader:
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        try:
            if scaler:
                with autocast():
                    outputs = model(inputs, attention_mask)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs, attention_mask)
                loss = criterion(outputs, labels)
        except ValueError as e:
            print(f"Forward Error: {e}")
            continue  # Skip this batch

        if scaler:
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    return epoch_loss, epoch_acc.item()


def validate(model, dataloader, criterion, device):
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    val_total_samples = 0

    with torch.no_grad():
        for inputs, attention_mask, labels in dataloader:
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            try:
                outputs = model(inputs, attention_mask)
                loss = criterion(outputs, labels)
            except ValueError as e:
                print(f"Forward Error: {e}")
                continue  # Skip this batch

            val_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels)
            val_total_samples += labels.size(0)

            # Memory management
            del inputs, attention_mask, labels, outputs, loss
            torch.cuda.empty_cache()

    val_epoch_loss = val_running_loss / val_total_samples
    val_epoch_acc = val_running_corrects.double() / val_total_samples

    return val_epoch_loss, val_epoch_acc.item()


# ============================================
# Main Execution Block
# ============================================

def main():
    # ============================================
    # Configuration
    # ============================================

    # Hyperparameters
    num_epochs = 20
    learning_rate = 1e-4
    batch_size = 32
    num_layers = 2
    nhead = 8
    d_model = 256
    dropout_rate = 0.3

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============================================
    # Data Preparation (Replace with Actual Data Loading)
    # ============================================

    ecg_data_A = load_and_preprocess_ecg_data(OFFSET, [2], dtype, DEBUG, CLUSTERING, PLOT)[:1500]
    # ecg_data_B = load_and_preprocess_ecg_data(OFFSET, [4], dtype, DEBUG, CLUSTERING, PLOT)

    # 샘플 레이블 생성
    label_to_idx_A = {label: idx for idx, label in enumerate(set(d['label'] for d in ecg_data_A))}
    data = [d['data'] for d in ecg_data_A]
    labels = np.array([label_to_idx_A[d['label']] for d in ecg_data_A])

    num_classes = len(label_to_idx_A)

    # 데이터 무결성 검사
    if not np.all(np.isfinite(labels)):
        raise ValueError("레이블에 비정상적인 값(NaN 또는 Inf)이 포함되어 있습니다.")

    # Split into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 클래스 가중치 무결성 검사
    if torch.isnan(class_weights_tensor).any() or torch.isinf(class_weights_tensor).any():
        raise ValueError("class_weights_tensor에 NaN 또는 INF 값이 포함되어 있습니다.")

    # ============================================
    # Dataset and DataLoader
    # ============================================

    train_dataset = ECGDataset(train_data, train_labels, augment=True)
    val_dataset = ECGDataset(val_data, val_labels, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # ============================================
    # Model, Loss Function, Optimizer, Scheduler
    # ============================================

    model = ECGTransformer(
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(device)

    # 최종 분류기를 위한 가중치 초기화
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    initialize_weights(model.fc)

    # Define loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    # ============================================
    # Training Loop with Early Stopping
    # ============================================

    best_val_acc = 0.0
    best_model_state = None
    patience = 5
    trigger_times = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            scaler,
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validate
        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            trigger_times = 0
            print("Validation accuracy improved. Saving best model.")
        else:
            trigger_times += 1
            print(f"No improvement in validation accuracy for {trigger_times} epochs.")

        # Early stopping
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

    # ============================================
    # Load Best Model
    # ============================================

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")

    # ============================================
    # Save the Best Model
    # ============================================

    torch.save(model.state_dict(), 'best_ecg_transformer_model.pth')
    print("Best model saved to 'best_ecg_transformer_model.pth'.")

    # ============================================
    # Evaluation and Visualization (Optional)
    # ============================================

    # Example: Plotting attention masks and predictions for a few samples
    model.eval()
    with torch.no_grad():
        for inputs, attention_mask, labels in val_loader:
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(inputs, attention_mask)
            _, preds = torch.max(outputs, 1)

            # Convert tensors to CPU and numpy
            inputs_np = inputs.cpu().numpy()
            attention_mask_np = attention_mask.cpu().numpy()
            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()

            # Plot the first sample in the batch
            sample_idx = 0
            sequence_length = attention_mask_np[sample_idx].sum()
            plt.figure(figsize=(12, 6))
            for i in range(sequence_length):
                plt.subplot(4, sequence_length // 4 + 1, i + 1)
                plt.imshow(inputs_np[sample_idx, i, 0, :, :], cmap='gray')
                plt.axis('off')
            plt.suptitle(f"True Label: {labels_np[sample_idx]}, Predicted: {preds_np[sample_idx]}")
            plt.show()

            break  # Plot only one batch


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    main()
