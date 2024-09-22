import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from itertools import chain
import torch
import torch.nn as nn
from transformers import BertModel
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import random
from scipy.ndimage import shift
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Constants
N_DATA = [2,4,6]  # 완료 2,4,6
OFFSET = None
DEBUG = False
LEARNING_RATE = 0.001
PLOT = False
dtype = 1
CLUSTERING = True

def augment_ecg(ecg_signal, method):
    if method == "amplitude_scaling":
        # 신호의 진폭을 일정 비율로 증폭/축소
        scaling_factor = random.uniform(0.8, 1.2)
        return ecg_signal * scaling_factor

    elif method == "add_noise":
        # 작은 Gaussian Noise 추가
        noise = np.random.normal(0, 0.01, ecg_signal.shape)
        return ecg_signal + noise

    elif method == "time_shift":
        # 신호를 시간 축에서 좌우로 이동
        shift_value = random.randint(-10, 10)
        return shift(ecg_signal, shift_value, cval=0)

    elif method == "random_crop":
        # 신호의 일부를 자르고, 나머지 부분을 반환
        crop_size = random.randint(10, 50)
        start_idx = random.randint(0, len(ecg_signal) - crop_size)
        return np.pad(ecg_signal[start_idx:], (0, start_idx), 'constant', constant_values=0)

    return ecg_signal


class ECGDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.augmentation_methods = ['amplitude_scaling', 'add_noise', 'time_shift', 'random_crop']

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        ecg_signal = self.data_list[idx]

        # 두 개의 다른 변형을 적용
        method1 = random.choice(self.augmentation_methods)
        method2 = random.choice(self.augmentation_methods)

        x_i = augment_ecg(ecg_signal, method1)
        x_j = augment_ecg(ecg_signal, method2)

        return torch.tensor(x_i, dtype=torch.float32), torch.tensor(x_j, dtype=torch.float32)


def train_simclr(model, dataloader, optimizer, criterion, epochs=10, device='cuda'):
    model.train()
    model.to(device)  # 모델을 GPU로 전송

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()

            # 데이터를 GPU로 전송
            x_i, x_j = data[0].to(device), data[1].to(device)

            # SimCLR 모델에 입력
            z_i = model(x_i)
            z_j = model(x_j)

            # Contrastive Loss 계산
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:  # 10번마다 loss 출력
                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item()}')

        print(f'Epoch {epoch + 1}, Average Loss: {total_loss / len(dataloader)}')

    # 학습 완료 후 모델 저장
    torch.save(model.state_dict(), 'simclr_model.pth')
    print("Model saved to simclr_model.pth")

class SimCLRModel(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLRModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=768, kernel_size=3, padding=1)
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension for Conv1D, shape: (batch_size, 1, sequence_length)
        x = self.conv1d(x)  # Apply 1D Convolution, shape: (batch_size, 768, sequence_length)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, sequence_length, 768)
        outputs = self.encoder(inputs_embeds=x)  # Use BERT encoder with inputs_embeds
        z = self.projector(outputs.pooler_output)  # Project the BERT output
        return z


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        # Normalize the embeddings
        z_i = nn.functional.normalize(z_i, dim=-1)
        z_j = nn.functional.normalize(z_j, dim=-1)

        # Compute similarity matrix
        similarity_matrix = self.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0))

        # Compute the loss
        labels = torch.arange(batch_size).to(z_i.device)
        loss = nn.CrossEntropyLoss()(similarity_matrix / self.temperature, labels)
        return loss

# 1. 데이터 로딩 및 전처리
def load_and_preprocess_ecg_data(offset, n_data, dtype, debug, clustering, plot):
    from analysis import load_and_preprocess_data
    from preprocess_utils import segment_ecg

    preprocessed_dataset = load_and_preprocess_data(offset, n_data, dtype, debug, clustering, plot)
    for item in preprocessed_dataset:
        item['data'] = np.asarray(item['data'])[:,0]#segment_ecg(item['data']))[:, :, 0]  # ECG 데이터를 분할
    return preprocessed_dataset


# 2. PCA 변환
def apply_pca(data_combined, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data_combined)


# 3. 클러스터링 분석 (Elbow Method & Silhouette Score)
def combined_clustering_analysis(data_combined):
    cluster_range = range(2, 10)
    sse = []  # Sum of Squared Errors
    silhouette_scores = []  # Silhouette Scores

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data_combined)

        sse.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(data_combined, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

    # Plot Elbow and Silhouette Score
    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('SSE', color='tab:blue')
    ax1.plot(cluster_range, sse, marker='o', color='tab:blue', label="SSE")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Silhouette Score', color='tab:green')
    ax2.plot(cluster_range, silhouette_scores, marker='o', color='tab:green', label="Silhouette Score")
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    plt.title("Elbow Method and Silhouette Score")
    plt.show()


# 4. DBSCAN 클러스터링 및 시각화
def dbscan_clustering(data_combined, data_2d):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(data_combined)

    # Check how many clusters DBSCAN found (-1 represents noise)
    unique_labels_dbscan = np.unique(dbscan_labels)
    n_clusters_dbscan = len(unique_labels_dbscan) - (1 if -1 in unique_labels_dbscan else 0)

    # Visualize DBSCAN results
    plt.figure(figsize=(8, 6))
    scatter_dbscan = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=dbscan_labels, cmap='viridis')

    plt.title(f"DBSCAN Clustering Visualization (Estimated Clusters: {n_clusters_dbscan})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()


# 7. t-SNE 또는 PCA를 사용한 시각화 함수 정의
def visualize_embeddings(embeddings, labels, method='tsne'):
    # 레이블이 리스트나 ndarray인 경우, 단일 값으로 변환
    if isinstance(labels[0], np.ndarray):
        labels = [str(label[0]) for label in labels]  # 각 레이블의 첫 번째 값을 문자열로 변환

    if method == 'tsne':
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)
    elif method == 'pca':
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

    # 시각화
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='viridis')
    plt.title(f'{method.upper()} Visualization of Embeddings')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.legend(title='Labels')
    plt.show()


def extract_and_visualize_embeddings(model, dataloader, method='tsne', device='cuda'):
    model.eval()  # 평가 모드로 전환
    model.to(device)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            x_i, labels = data[0].to(device), data[1]  # 데이터 및 레이블 가져오기
            embeddings = model(x_i).cpu().numpy()  # 학습된 임베딩 추출
            all_embeddings.append(embeddings)
            all_labels.extend(labels.numpy())  # 레이블 저장

    all_embeddings = np.vstack(all_embeddings)  # 임베딩을 하나로 결합
    visualize_embeddings(all_embeddings, all_labels, method=method)  # 시각화


# 5. Main function
def main():
    base_encoder = BertModel.from_pretrained('bert-base-uncased')
    model = SimCLRModel(base_encoder)

    # Step 1: 데이터 로딩 및 전처리
    preprocessed_dataset = load_and_preprocess_ecg_data(OFFSET, N_DATA, dtype, DEBUG, CLUSTERING, PLOT)

    # 데이터 결합 및 라벨 준비
    data_combined = np.vstack([entry['data'] for entry in preprocessed_dataset])
    labels_combined = list(chain(*[[entry['label']] * len(entry['data']) for entry in preprocessed_dataset]))

    # time_series_tensor = torch.tensor(data_combined, dtype=torch.float32).unsqueeze(2)

    model.load_state_dict(torch.load('simclr_model.pth'))

       #
    dataset = ECGDataset(data_combined)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    extract_and_visualize_embeddings(model, dataloader, method='tsne', device='cuda')

    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # criterion = ContrastiveLoss()
    # train_simclr(model, dataloader, optimizer, criterion, epochs=10, device='cuda')

    # # Step 2: PCA 적용
    # data_2d = apply_pca(data_combined)
    #
    # # Step 3: 클러스터링 분석 (Elbow Method와 Silhouette Score)
    # # combined_clustering_analysis(data_combined)
    #
    # # Step 4: DBSCAN 클러스터링 및 시각화
    # print("dbscan_clustering")
    # dbscan_clustering(data_combined, data_2d)


if __name__ == "__main__":
    main()
