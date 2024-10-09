
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from stockwell import st
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import logging
import itertools
import json  # 분류 보고서 저장을 위한 임포트
from collections import Counter
# Additional imports for Efficient Transformer, DropBlock, and Optimizers
from linformer import Linformer  # Efficient Transformer
from dropblock import DropBlock2D  # DropBlock
from ranger import Ranger  # Ranger optimizer
from torch_optimizer import RAdam  # RAdam optimizer
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from tqdm import tqdm  # tqdm 임포트

# 로깅 설정
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# 재현성을 위한 시드 설정
def set_seed(seed):
    torch.manual_seed(seed)                   # CPU
    torch.cuda.manual_seed(seed)              # GPU
    torch.cuda.manual_seed_all(seed)          # 모든 GPU
    np.random.seed(seed)                      # Numpy
    random.seed(seed)                         # Python random
    torch.backends.cudnn.deterministic = True # 결정론적 결과
    torch.backends.cudnn.benchmark = False    # 재현성을 위해 benchmark 비활성화

# Stockwell 변환 함수
def stockwell_transform(signal, fmin, fmax, signal_length):
    df = 1. / signal_length
    fmin_samples = int(fmin / df)
    fmax_samples = int(fmax / df)
    trans_signal = st.st(signal, fmin_samples, fmax_samples)
    return trans_signal

# 파일을 전처리하고 저장하는 함수 (병렬 처리 버전)
def preprocess_file(args):
    file_path, save_dir, fmin, fmax, signal_length = args
    try:
        # 신호 읽기
        signal = pd.read_csv(file_path, header=None).squeeze().to_numpy()

        # Stockwell 변환 적용
        trans_signal = stockwell_transform(signal, fmin, fmax, signal_length)

        # 실수부와 허수부 분리
        real_part = np.real(trans_signal)
        imag_part = np.imag(trans_signal)

        # 스택하고 저장
        transformed_signal = np.stack((real_part, imag_part), axis=0)
        filename = os.path.basename(file_path).replace('.csv', '_transformed.npy')
        save_path = os.path.join(save_dir, filename)
        np.save(save_path, transformed_signal)
    except Exception as e:
        logging.error(f"{file_path} 처리 중 오류 발생: {e}")

def preprocess_and_save(file_paths, save_dir, fmin, fmax, signal_length):
    os.makedirs(save_dir, exist_ok=True)
    args_list = [(file_path, save_dir, fmin, fmax, signal_length) for file_path in file_paths]
    num_processes = max(1, cpu_count() - 1)  # 한 개의 CPU 코어는 남겨둠

    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(preprocess_file, args_list), total=len(args_list)))

# 커스텀 데이터셋 클래스
class CustomECGDataset(Dataset):
    def __init__(self, file_paths, labels, transform_dir):
        self.file_paths = file_paths
        self.labels = labels
        self.transform_dir = transform_dir

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            # 전처리된 데이터 로드
            original_file = os.path.basename(self.file_paths[idx]).replace('.csv', '_transformed.npy')
            transformed_path = os.path.join(self.transform_dir, original_file)
            signal = np.load(transformed_path)
            signal_tensor = torch.tensor(signal, dtype=torch.float32)

            # 레이블
            label = self.labels[idx]
            return signal_tensor, label
        except Exception as e:
            logging.error(f"인덱스 {idx}에서 데이터 로드 중 오류 발생: {e}")
            # 오류 발생 시 0 텐서와 기본 레이블 반환
            signal_tensor = torch.zeros((2, 151, 1000), dtype=torch.float32)  # 필요한 경우 차원 조정
            label = 0
            return signal_tensor, label

# Efficient Transformer (Linformer) 클래스
class EfficientTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, max_seq_len=256, k=256):
        super(EfficientTransformer, self).__init__()
        self.linformer = Linformer(
            dim=d_model,
            seq_len=max_seq_len,
            depth=num_layers,
            heads=nhead,
            k=k,
            one_kv_head=True,
            share_kv=True,
            reversible=False,
            # layer_norm_epsilon=1e-5,
            dropout=dropout,
            # attention_dropout=dropout
        )

    def forward(self, x):
        return self.linformer(x)

# DropBlock을 포함한 TemporalBlock2D
class TemporalBlock2D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2, use_batchnorm=False, use_dropblock=False, dropblock_size=7):
        super(TemporalBlock2D, self).__init__()
        # "same" 패딩 계산
        pad_height = (kernel_size[0] - 1) * dilation[0] // 2
        pad_width = (kernel_size[1] - 1) * dilation[1] // 2
        padding = (pad_width, pad_width, pad_height, pad_height)  # (좌, 우, 상, 하)

        # 첫 번째 합성곱 층
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=0, dilation=dilation)
        self.batchnorm1 = nn.BatchNorm2d(n_outputs) if use_batchnorm else nn.Identity()
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)  # LeakyReLU 사용
        self.dropout1 = nn.Dropout(dropout)
        self.dropblock1 = DropBlock2D(block_size=dropblock_size, drop_prob=dropout) if use_dropblock else nn.Identity()

        # 두 번째 합성곱 층
        self.conv2 = nn.Conv2d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=0, dilation=dilation)
        self.batchnorm2 = nn.BatchNorm2d(n_outputs) if use_batchnorm else nn.Identity()
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)  # LeakyReLU 사용
        self.dropout2 = nn.Dropout(dropout)
        self.dropblock2 = DropBlock2D(block_size=dropblock_size, drop_prob=dropout) if use_dropblock else nn.Identity()

        # 잔차 연결
        self.downsample = nn.Conv2d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        # 패딩 레이어
        self.pad = nn.ZeroPad2d(padding)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv1(out)
        out = self.batchnorm1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.dropblock1(out)

        out = self.pad(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.dropblock2(out)

        # 잔차 연결
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# 모델 정의
class TemporalConvNet2D(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=(3, 3), dropout=0.2,
                 num_classes=3, use_batchnorm=False, use_transformer=False,
                 transformer_layers=4, transformer_heads=8, max_seq_len=256, use_dropblock=False, dropblock_size=7):
        super(TemporalConvNet2D, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = (2 ** i, 2 ** i)
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock2D(in_channels, out_channels, kernel_size,
                                       stride=1, dilation=dilation_size, dropout=dropout,
                                       use_batchnorm=use_batchnorm, use_dropblock=use_dropblock, dropblock_size=dropblock_size)]

        self.network = nn.Sequential(*layers)

        # Adaptive Pooling 레이어 추가
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))  # 원하는 크기로 설정

        if use_transformer:
            # Efficient Transformer 사용
            self.transformer = EfficientTransformer(
                d_model=num_channels[-1],
                nhead=transformer_heads,
                num_layers=transformer_layers,
                dim_feedforward=2048,
                dropout=dropout,
                max_seq_len=256,
                k=256  # Linformer specific parameter
            )
        else:
            self.transformer = None

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels[-1], num_classes)  # Use num_channels[-1] as in_channels

    def forward(self, x):
        x = self.network(x)
        x = self.adaptive_pool(x)  # Adaptive Pooling 적용으로 크기 감소

        if self.transformer is not None:
            b, c, h, w = x.size()
            x = x.view(b, c, h * w).permute(0, 2, 1)  # (batch_size, seq_len, embedding_dim)
            x = self.transformer(x)  # Efficient Transformer
            x = x.mean(dim=1)  # 시퀀스 차원에 대해 평균
        else:
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x
# Hook 함수 정의
def generate_cam_and_visualize(model, data_tensor):
    activation = {}
    label =["N" ,"S" ,"V" ,"Q" ,"F"]
    # Conv 레이어에 Hook 설정
    def hook_fn(module, input, output):
        activation['conv_output'] = output.detach()

    # 마지막 Conv 레이어에 hook을 걸기
    model.network[2].conv2.register_forward_hook(hook_fn)

    # 모델에 데이터를 통과시키기
    model.eval()
    output = model(data_tensor)

    # 예측된 클래스
    pred_class = output.argmax(dim=1).item()
    print(pred_class)
    # Conv 레이어의 출력 가져오기 (실수부와 허수부가 각각 따로 있음)
    conv_output = activation['conv_output'].squeeze().cpu().numpy()
    # print(conv_output.shape)
    # FC 레이어의 가중치 가져오기
    params = list(model.fc.parameters())
    weight_softmax = params[0].cpu().data.numpy()

    cam = np.zeros(conv_output.shape[1:])  # (151, 1000) 크기의 CAM 생성
    for i in range(conv_output.shape[0]):  # 각 채널에 대해
        cam += weight_softmax[pred_class, i] * conv_output[i]  # 가중치를 곱해서 더함

    # CAM을 정규화
    data_tensor =data_tensor.squeeze()
    t = np.linspace(0, 10, 1000)
    extent =(t[0] ,t[-1] ,0 ,15)
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # 정규화
    real_part = data_tensor[0].cpu().numpy()
    imag_part = data_tensor[1].cpu().numpy()

    abs_val = np.sqrt(real_part**2 + imag_part**2)

    # plt.imshow(abs_val, origin='lower', extent=extent)
    # plt.show()

    # plt.imshow(cam, cmap='jet', alpha=0.5)
    # plt.colorbar()
    # plt.show()
    plt.figure(figsize=(12, 6))
    extent = (0, abs_val.shape[1], 0, abs_val.shape[0])

    cam_img =abs_val *cam
    plt.imshow(cam_img, aspect='auto', origin='lower', cmap='jet', alpha=0.5, extent=extent)
    plt.title('MIT-BIH Data Label:  ' +label[pred_class ] +" CAM")

    plt.tight_layout()
    plt.show()


def load_npy_data(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(file_path)
    tensor_data = torch.tensor(data, dtype=torch.float32)
    tensor_data = tensor_data.unsqueeze(0).to(device)
    return tensor_data


# 실험 실행 함수
def run_experiment(config):
    # 로그 파일 설정
    file_dict = {"N": 0, "S": 1, "V": 2 ,"Q" :3 ,"F" :4}
    file_paths = []
    labels = []
    # 시드 설정
    seed = config.get('seed', 42)  # 고정된 시드 값 사용
    set_seed(seed)

    # 파라미터 설정
    input_size = config['input_size']
    output_size = config['output_size']
    num_channels = config['num_channels']
    kernel_size = config['kernel_size']
    dropout = config['dropout']
    fmin = config['fmin']
    fmax = config['fmax']
    signal_length = config['signal_length']
    target_length = config['target_length']
    use_batchnorm = config['use_batchnorm']
    use_transformer = config['use_transformer']
    transformer_layers = config.get('transformer_layers', 4)
    transformer_heads = config.get('transformer_heads', 8)
    batch_size = config['batch_size']
    accumulation_steps = config['accumulation_steps']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    epochs = config['epochs']
    patience = config['patience']
    root_path = config['root_path']
    preprocessed_dir = config['preprocessed_dir']
    augment = config.get('augment', False)  # 데이터 증강 여부
    use_dropblock = config.get('use_dropblock', True)  # DropBlock 사용 여부
    dropblock_size = config.get('dropblock_size', 7)  # DropBlock 크기
    optimizer_type = config.get('optimizer', 'Ranger')  # 옵티마이저 선택: 'RAdam' 또는 'Ranger'

    for folder_name, label in file_dict.items():
        folder_path = os.path.join(root_path, folder_name)
        if not os.path.isdir(folder_path):
            logging.warning(f"폴더 {folder_path} 가 존재하지 않습니다.")
            continue
        for file in os.listdir(folder_path)[:15000]:  # Changed to 1000 as per user's last code
            if file.endswith('.csv'):
                file_paths.append(os.path.join(folder_path, file))
                labels.append(label)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalConvNet2D(
        num_inputs=input_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        num_classes=output_size,
        use_batchnorm=use_batchnorm,
        use_transformer=use_transformer,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        max_seq_len=256,
        use_dropblock=use_dropblock,
        dropblock_size=dropblock_size
    )
    model =model.to(device)

    pth_file_path ="./models/exp_nc3_bnTrue_tfTrue_do0.1_lr0.0005_tl6_th16_db7_optRAdam_best.pth"
    checkpoint = torch.load(pth_file_path)  # 전체 체크포인트 로드
    model.load_state_dict(checkpoint['model_state_dict'])  # 모델 파라미터만 로드
    model.eval()
    print(model)

    for i in ["N" ,"S" ,"V" ,"Q" ,"F"]:
        for j in range(20 ,25):
            file_path = './MIT_npy/smote_ ' + i +'_ ' +str(j ) +'_transformed.npy'
            data_tensor = load_npy_data(file_path)
            # 모델을 사용하여 CAM 생성 및 시각화
            generate_cam_and_visualize(model, data_tensor)

def generate_experiment_configs():
    # 하이퍼파라미터 그리드 정의
    num_channels_options = [
        [32, 64, 128],
        # [64, 64, 128],
        # [16, 32],
        # [64, 128,  256],
        # [32, 64, 128, 256]
    ]
    use_batchnorm_options = [True]  # , False]
    use_transformer_options = [True]  # , False]
    dropout_options = [0.1]  # , 0.2, 0.3]
    learning_rate_options = [0.0005]  # [0.0001, 0.0005, 0.001]
    # transformer_layers_options = [4, 6]  # 추가 실험: Transformer 레이어 수
    transformer_layers_options = [6]
    # transformer_heads_options = [8, 16]  # 추가 실험: Transformer 헤드 수
    transformer_heads_options = [16]
    # seed_options 제거됨

    # DropBlock options
    use_dropblock_options = [False]
    dropblock_size_options = [7]  # [5, 7]

    # Optimizer options
    optimizer_options = ['RAdam']  # 'RAdam' 또는 'Ranger'

    # 다른 하이퍼파라미터는 고정값 또는 필요에 따라 추가
    base_config = {
        'experiment_name': 'experiment',
        'input_size': 2,
        'output_size': 5,
        'kernel_size': (3, 3),
        'fmin': 0,
        'fmax': 15,
        'signal_length': 10,
        'target_length': 151,  # 고정된 신호 길이
        'batch_size': 16,       # 배치 크기 감소
        'accumulation_steps': 4,
        'weight_decay': 1e-5,
        'epochs': 50,
        'patience': 8,
        'root_path': './smote_augmented_data_by_label',   # 실제 데이터 경로로 수정 필요
        'preprocessed_dir': './MIT_npy', # 실제 저장 경로로 수정 필요
        'seed': 5148,  # 고정된 시드 값 설정
        'augment': False,  # 데이터 증강 여부 (비활성화 as per earlier)
        'use_dropblock': True,  # DropBlock 사용 여부
        'dropblock_size': 7  # DropBlock 크기
    }

    # 하이퍼파라미터 조합 생성
    configs = []
    for num_channels, use_batchnorm, use_transformer, dropout, learning_rate, transformer_layers, transformer_heads, use_dropblock, dropblock_size, optimizer in itertools.product(
            num_channels_options, use_batchnorm_options, use_transformer_options,
            dropout_options, learning_rate_options, transformer_layers_options, transformer_heads_options,
            use_dropblock_options, dropblock_size_options, optimizer_options
    ):
        config = base_config.copy()
        config['num_channels'] = num_channels
        config['use_batchnorm'] = use_batchnorm
        config['use_transformer'] = use_transformer
        config['dropout'] = dropout
        config['learning_rate'] = learning_rate
        config['transformer_layers'] = transformer_layers
        config['transformer_heads'] = transformer_heads
        config['use_dropblock'] = use_dropblock
        config['dropblock_size'] = dropblock_size
        config['optimizer'] = optimizer
        # seed_options 제거로 인해 seed는 base_config에서 가져옵니다.
        config['experiment_name'] = f"exp_nc{len(num_channels)}_bn{use_batchnorm}_tf{use_transformer}_do{dropout}_lr{learning_rate}_tl{transformer_layers}_th{transformer_heads}_db{dropblock_size}_opt{optimizer}"
        configs.append(config)

    return configs

# 메인 실행 부분
if __name__ == "__main__":
    configs = generate_experiment_configs()
    for config in configs:
        run_experiment(config)