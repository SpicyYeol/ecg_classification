import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from ECGDataset import ECGDataset
from data_source_utils import load_dataset
from models.CNN2D import CNN2D
from models.DenseNet import densenet_cifar
from models.ECGCNN import ECGCNN
from models.LSTMNet import LSTMNet
from models.ResNet import resnet18
from models.ResNet1D import resnet1d18
from models.TemporalConvNet import TemporalConvNet
from models.TransformerModel import TransformerModel
from models.VGGNet import VGGNet
from models.ViT import ViT
from preprocess_utils import preprocess_dataset
import logging
import matplotlib.pyplot as plt
import os

# 로그 설정
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Configuration parameters
N_DATA = [1]
OFFSET = None
DEBUG = True
MODEL_TYPES = [1]#, 2, 3, 4, 11, 12, 13, 14]#, 15, 16]
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PLOT = False
SAVE_CAM = True  # Save CAM images
CAM_DIR = 'cam_images'  # Directory to save CAM images
os.makedirs(CAM_DIR, exist_ok=True)
'''
    1. ECGCNN (1DCNN)           1D
    2. resnet1d18               1D
    3. LSTMNet (LSTM)           1D
    4. TemporalConvNet (TCN )   1D
    11. CNN2D (2DCNN)           2D
    12. VGGNet                  2D
    13. ResNet                  2D
    14. DenseNet                2D
    15. TransformerModel        2D
    16. ViT                     2D
'''

class WeightedMSELoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedMSELoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        loss = (self.class_weights * (inputs - targets) ** 2).mean()
        return loss



def compute_class_weights(data_list, num_classes):
    labels = np.vstack([item['labels'] for item in data_list])
    label_counts = labels.sum(axis=0)

    # 0으로 나누는 경우를 피하기 위해 작은 값을 추가
    epsilon = 1e-6
    small_class_indices = label_counts <= epsilon
    label_counts[small_class_indices] = epsilon

    class_weights = 1.0 / label_counts
    class_weights[small_class_indices] = 0  # epsilon인 경우 가중치를 0으로 설정
    class_weights = class_weights / class_weights.sum() * num_classes  # normalize weights
    return torch.tensor(class_weights, dtype=torch.float32)

def load_and_preprocess_data(offset, n_data, dtype, debug,plot=False):
    preprocessed_dataset = []
    for n in n_data:
        dataset = load_dataset(offset=offset, n_data=n)
        if len(dataset) > 0:
            for data in dataset:
                preprocessed_dataset.append(preprocess_dataset(data['data'], dtype,data['fs'], plot=plot, debug=debug))
                if preprocessed_dataset is not None and not debug:
                    print("Stacked Contents Shape:", preprocessed_dataset.shape)

                    # Save the preprocessed dataset to a dynamically named file
                    file_name = f'n_{n}.py'
                    with open(file_name, 'w') as file:
                        file.write(f'preprocessed_dataset = {repr(preprocessed_dataset.tolist())}')

            return preprocessed_dataset
        else:
            print("No valid ECG data found in the directory.")
            continue
            #return None


def create_dataloaders(preprocessed_dataset, batch_size=32, split_ratio=0.8):
    ecg_dataset = ECGDataset(preprocessed_dataset)
    train_size = int(split_ratio * len(ecg_dataset))
    val_size = len(ecg_dataset) - train_size
    train_dataset, val_dataset = random_split(ecg_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def initialize_model(model_type, ecg_data_length, num_classes):
    if model_type == 1:
        return ECGCNN(num_classes=num_classes, ecg_data_length=ecg_data_length)
    elif model_type == 2:
        return resnet1d18(num_classes=num_classes)
    elif model_type == 3:
        return LSTMNet(num_classes=num_classes)
    elif model_type == 4:
        num_channels = [16, 32, 64]
        return TemporalConvNet(num_inputs=3, num_channels=num_channels, num_classes=num_classes, kernel_size=3,
                               dropout=0.2)
    elif model_type == 11:
        return CNN2D(num_classes=num_classes)
    elif model_type == 12:
        return VGGNet(num_classes=num_classes)
    elif model_type == 13:
        return resnet18(num_classes=num_classes)
    elif model_type == 14:
        return densenet_cifar(num_classes=num_classes)
    elif model_type == 15:
        return TransformerModel(num_classes=num_classes)
    elif model_type == 16:
        img_size = 32
        patch_size = 4
        in_channels = 3
        embed_dim = 128
        depth = 6
        num_heads = 8
        mlp_dim = 256
        dropout = 0.1
        return ViT(img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, dropout)


def generate_cam(model, ecg_data, target_layer, model_type, num_classes):
    # Ensure the input tensor requires gradient
    ecg_data.requires_grad_()

    # Hook for the gradients
    gradients = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Hook for the activations
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    handle_backward = target_layer.register_backward_hook(backward_hook)
    handle_forward = target_layer.register_forward_hook(forward_hook)

    # Forward pass
    if type(model).__name__ == 'LSTMNet':
        output = model(torch.permute(ecg_data, (0, 2, 1)))
    else:
        output = model(ecg_data)

    cams = []
    for i in range(num_classes):
        model.zero_grad()
        if output.grad is not None:
            output.grad.zero_()
        loss = output[:, i].sum()
        loss.backward(retain_graph=True)

        grad = gradients[-1].cpu().data.numpy()
        act = activations[-1].cpu().data.numpy()

        # Calculate the weights
        weights = np.mean(grad, axis=2, keepdims=True)

        # Calculate the CAM
        cam = np.sum(weights * act, axis=1)
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cams.append(np.uint8(cam * 255))

        # Clear gradients for the next iteration
        gradients.clear()

    handle_backward.remove()
    handle_forward.remove()

    return cams


def save_cam_image(cams, ecg_data, model_type, epoch, idx, num_classes):
    for i in range(num_classes):
        plt.figure(figsize=(10, 5))

        # Plot the ECG data
        ecg_array = ecg_data[0][0].cpu().detach().numpy()
        plt.plot(ecg_array, color='black', alpha=0.6, label='ECG Data')

        # Create CAM overlay
        cam_overlay = cams[i][0]

        # Normalize CAM overlay for visualization
        cam_overlay = (cam_overlay - cam_overlay.min()) / (cam_overlay.max() - cam_overlay.min())

        # Overlay CAM on ECG data
        extent = [0, len(ecg_array), ecg_array.min(), ecg_array.max()]
        plt.imshow(cam_overlay.reshape(1, -1), cmap='jet', aspect='auto', alpha=0.5, extent=extent)
        plt.colorbar(label='CAM Intensity')
        plt.title(f'CAM Overlay for Class {i}')
        plt.legend()

        # Use index instead of labels for filename
        plt.savefig(os.path.join(CAM_DIR, f'model_{model_type}_epoch_{epoch}_sample_{idx}_class_{i}.png'))
        plt.close()


def train_and_evaluate(model, train_loader, val_loader, class_weights, num_epochs=10, lr=0.001, weighted=False, model_type=None):
    if not weighted:
        criterion = nn.MSELoss()
    else:
        criterion = WeightedMSELoss(class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    target_layer = None
    if model_type == 1:  # For ECGCNN
        target_layer = model.conv3

    for epoch in range(num_epochs):
        model.train()
        for ecg_data, labels in train_loader:
            ecg_data = ecg_data.to(next(model.parameters()).device)  # Ensure ecg_data is on the correct device
            labels = labels.to(next(model.parameters()).device)      # Ensure labels are on the correct device

            if type(model).__name__ == 'LSTMNet':
                outputs = model(torch.permute(ecg_data, (0, 2, 1)))
            else:
                outputs = model(ecg_data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        all_labels = []
        all_outputs = []
        for idx, (ecg_data, labels) in enumerate(val_loader):
            ecg_data = ecg_data.to(next(model.parameters()).device)  # Ensure ecg_data is on the correct device
            labels = labels.to(next(model.parameters()).device)      # Ensure labels are on the correct device

            if type(model).__name__ == 'LSTMNet':
                outputs = model(torch.permute(ecg_data, (0, 2, 1)))
            else:
                outputs = model(ecg_data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().detach().numpy())

            # Generate and save CAM
            if SAVE_CAM and target_layer and epoch == 9:
                ecg_data.requires_grad_()  # Ensure requires_grad is True before generating CAM
                cams = generate_cam(model, ecg_data, target_layer, model_type, labels.shape[1])
                save_cam_image(cams, ecg_data, model_type, epoch, idx, labels.shape[1])

        val_loss /= len(val_loader)
        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)
        r2 = r2_score(all_labels, all_outputs)

        label_r2_scores = []
        label_mse_scores = []
        for i in range(all_labels.shape[1]):
            label_r2 = r2_score(all_labels[:, i], all_outputs[:, i])
            label_mse = mean_squared_error(all_labels[:, i], all_outputs[:, i])
            label_r2_scores.append(label_r2)
            label_mse_scores.append(label_mse)
            logger.info(f'Label {i} R2 Score: {label_r2:.4f}, MSE: {label_mse:.4f}')

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, R2 Score: {r2:.4f}')
        logger.info(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, R2 Score: {r2:.4f}')


def main():
    # Configuration parameters
    logger.info(f'Dataset Number: {N_DATA}')
    logger.info(f'Number of sampe data: {OFFSET}')
    logger.info(f'Mode: {DEBUG}')
    logger.info(f'Test Model lists: {MODEL_TYPES}')
    logger.info(f'NUM_EPOCHS: {NUM_EPOCHS}')
    logger.info(f'BATCH_SIZE: {BATCH_SIZE}')
    logger.info(f'LEARNING_RATE: {LEARNING_RATE}')

    print(f'Dataset Number: {N_DATA}')
    print(f'Number of sampe data: {OFFSET}')
    print(f'Mode: {DEBUG}')
    print(f'Test Model lists: {MODEL_TYPES}')
    print(f'NUM_EPOCHS: {NUM_EPOCHS}')
    print(f'BATCH_SIZE: {BATCH_SIZE}')
    print(f'LEARNING_RATE: {LEARNING_RATE}')


    preprocessed_dataset = None
    current_dtype = None

    for model_type in MODEL_TYPES:
        print(f'\nEvaluating Model Type: {model_type}')
        logger.info(f'Evaluating Model Type: {model_type}')
        dtype = 1 if model_type <= 10 else 2

        # dtype이 변경되었을 때만 데이터를 전처리합니다.
        if current_dtype != dtype:
            preprocessed_dataset = load_and_preprocess_data(OFFSET, N_DATA, dtype, DEBUG, PLOT)
            current_dtype = dtype

        if np.shape(preprocessed_dataset)[0] >= 1:
            train_loader, val_loader = create_dataloaders(preprocessed_dataset, BATCH_SIZE)

            ecg_data_length = preprocessed_dataset[0]['ecg'].shape[0]
            num_classes = preprocessed_dataset[0]['labels'].shape[0]

            class_weights = compute_class_weights(preprocessed_dataset, num_classes)
            for weighted_flag in [True]:#, True]:
                logger.info(f'weighted Loss Type: {weighted_flag}')
                model = initialize_model(model_type, ecg_data_length, num_classes)
                train_and_evaluate(model, train_loader, val_loader, class_weights, NUM_EPOCHS, LEARNING_RATE, weighted_flag, model_type)


if __name__ == "__main__":
    main()
