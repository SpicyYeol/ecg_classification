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
d_model = 64  # Transformer 모델 차원
nhead = 8  # Multi-head attention
num_layers = 4  # Transformer 레이어 수
dim_feedforward = 128  # FFN의 차원
seq_length = 300  # ECG 시퀀스 길이
batch_size = 32
num_epochs = 10
# num_classes = 4  # a, b, c, d에 대한 4개 클래스

# option = ['TRAIN', 'VAL', 'HEATMAP']
option = ['VAL']


# Define the Dataset class
class ECGDataset(Dataset):
    def __init__(self, data, labels):
        """
        data: list of numpy arrays, each of shape (N_i, 300)
        labels: numpy array of shape (Batch,)
        """
        self.data = data  # List of arrays of shape (N_i, 300)
        self.labels = labels  # Shape: (Batch,)

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


# Define Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# Define Spatial Attention Mechanism
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_out, mean_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


# Define the Feature Extraction module
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = DepthwiseSeparableConv(4, 32, kernel_size=3, padding=1)
        self.attention1 = SpatialAttention()
        self.conv2 = DepthwiseSeparableConv(32, 64, kernel_size=3, padding=1)
        self.attention2 = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        x = self.conv1(x)
        x = self.attention1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.attention2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        return x  # Output shape: (batch_size, channels, 16, 16)


# Define Custom Transformer Encoder Layer to extract attention weights
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, src, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask, need_weights=True)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src, attn_weights


# Define Custom Transformer Encoder
class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([CustomTransformerEncoderLayer(d_model, nhead) for _ in range(num_layers)])

    def forward(self, src, src_key_padding_mask=None):
        attn_weights_list = []
        output = src
        for mod in self.layers:
            output, attn_weights = mod(output, src_key_padding_mask=src_key_padding_mask)
            attn_weights_list.append(attn_weights)
        return output, attn_weights_list


# Define the ECG Transformer Model
class ECGTransformer(nn.Module):
    def __init__(self, num_classes):
        super(ECGTransformer, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.d_model = 64  # Must match the output channels from the feature extractor
        self.transformer_encoder = CustomTransformerEncoder(d_model=self.d_model, nhead=8, num_layers=6)
        self.fc = nn.Linear(self.d_model, num_classes)

    def forward(self, x, attention_mask):
        """
        x: tensor of shape (batch_size, max_length, 3, image_size, image_size)
        attention_mask: tensor of shape (batch_size, max_length), where 1 indicates valid positions
        """
        batch_size, max_length, C, H, W = x.size()
        x = x.view(batch_size * max_length, C, H, W)  # Merge batch and sequence dimensions
        x = self.feature_extractor(x)  # Output shape: (batch_size * max_length, d_model, 16, 16)
        x = x.view(batch_size, max_length, self.d_model, -1)  # Shape: (batch_size, max_length, d_model, 16*16)
        x = x.mean(-1)  # Global average pooling over spatial dimensions: (batch_size, max_length, d_model)

        # Transformer expects input of shape (batch_size, seq_length, d_model)
        # Prepare attention mask for Transformer
        # In PyTorch's MultiheadAttention, key_padding_mask should be True for positions that should be masked
        src_key_padding_mask = ~attention_mask  # Invert mask: True for padding positions

        # Pass through Transformer Encoder
        output, attn_weights_list = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Classification head
        # We need to aggregate the output taking into account the valid positions
        # We'll compute a weighted average of the outputs
        attention_mask = attention_mask.unsqueeze(-1)  # Shape: (batch_size, max_length, 1)
        masked_output = output * attention_mask  # Zero out the padding positions
        sum_output = masked_output.sum(dim=1)  # Sum over sequence length
        valid_lengths = attention_mask.sum(dim=1)  # Sum of valid positions
        avg_output = sum_output / valid_lengths  # Average over valid positions

        logits = self.fc(avg_output)  # Shape: (batch_size, num_classes)
        return logits, attn_weights_list


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

# Create dataset and DataLoader
train_dataset = ECGDataset(train_data, train_labels)
val_dataset = ECGDataset(val_data, val_labels)

train_loader  = DataLoader(train_dataset, batch_size=Batch, shuffle=True, collate_fn=collate_fn)
val_loader  = DataLoader(val_dataset, batch_size=Batch, shuffle=True, collate_fn=collate_fn)

# Initialize the model, loss function, and optimizer
num_classes = len(label_to_idx_A)  # Adjust based on your classification task
model = ECGTransformer(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop with model saving after every epoch
save_path = './ecg_transformer_model.pth'  # 모델을 저장할 경로

if os.path.exists(save_path):
    print(f"Loading model from {save_path}")
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded. Resuming training from epoch {start_epoch}.")
else:
    print("No saved model found. Starting training from scratch.")
    start_epoch = 0  # 모델을 처음부터 학습

if 'TRAIN' in option:
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        accumulation_steps = 8  # 4개의 미니 배치마다 한 번의 역전파
        for i, (inputs, attention_mask, labels) in enumerate(train_loader):
            outputs, _ = model(inputs, attention_mask)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)
            total_samples += labels.size(0)

            if (i + 1) % accumulation_steps == 0:
                print(loss.item())
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # 모델 저장
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        print(f"Model saved after epoch {epoch + 1}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0
        with torch.no_grad():
            for inputs, attention_mask, labels in val_loader:
                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outputs, _ = model(inputs, attention_mask)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                # Compute accuracy
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels)
                val_total_samples += labels.size(0)
        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_running_corrects.double() / val_total_samples
        print(f"Validation Loss: {val_epoch_loss:.4f}")
        print(f"Validation Accuracy: {val_epoch_acc:.4f}")

if 'VAL' in option:
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    val_total_samples = 0
    with torch.no_grad():
        for inputs, attention_mask, labels in val_loader:
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs, _ = model(inputs, attention_mask)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)

            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels)
            val_total_samples += labels.size(0)
    val_epoch_loss = val_running_loss / val_total_samples
    val_epoch_acc = val_running_corrects.double() / val_total_samples
    print(f"Validation Accuracy: {val_epoch_acc:.4f}")

if 'HEATMAP' in option:
    # Inference and Attention Map Visualization
    model.eval()
    with torch.no_grad():
        for inputs, attention_mask, labels in val_loader:
            outputs, attn_weights_list = model(inputs, attention_mask)
            # attn_weights_list: List of attention weights from each layer
            # Shape of each attn_weights: (batch_size, num_heads, max_length, max_length)
            # We need to consider the attention mask when interpreting attention weights

            # Stack attention weights from all layers
            attn_weights = torch.stack(
                attn_weights_list)  # Shape: (num_layers, batch_size, num_heads, max_length, max_length)
            # Average over heads and layers
            # attn_weights = attn_weights.mean(dim=2).mean(dim=0)  # Shape: (batch_size, max_length, max_length)
            attn_weights = attn_weights.mean(dim=0)  # Shape: (batch_size, max_length, max_length)

            for i in range(inputs.size(0)):  # Iterate over batch
                sample_attn = attn_weights[i]  # Shape: (max_length, max_length)
                valid_length = attention_mask[i].sum().item()
                N_i = int(valid_length)
                # Extract the valid part of the attention weights
                sample_attn_valid = sample_attn[:N_i, :N_i]  # Shape: (N_i, N_i)
                # Aggregate attention weights over query dimension
                attention_map = sample_attn_valid.mean(dim=0)  # Shape: (N_i,)
                # Normalize attention map
                attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
                # Convert to numpy for visualization
                attention_map_np = attention_map.cpu().numpy()
                # Here you can visualize the attention map
                # For example:

                import matplotlib.pyplot as plt
                plt.bar(range(N_i), attention_map_np)
                plt.xlabel('ECG Sequence Index')
                plt.ylabel('Attention Score')
                plt.title('Attention Map for Sample {} with label {}'.format(i,labels[i].cpu().numpy()))
                plt.show()
                # This attention map shows which ECG sequences are contributing to the classification

