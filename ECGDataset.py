import torch
from torch.utils.data import Dataset

from preprocess_utils import segment_ecg


class ECGDataset(Dataset):
    def __init__(self, preprocessed_dataset):
        """
        ECG 데이터를 PyTorch Dataset으로 변환하는 클래스.
        각 dict의 'data'와 'label' 필드를 쌍으로 반환.

        :param preprocessed_dataset: 분할된 ECG 데이터가 포함된 dict 리스트
        """
        self.data_label_pairs = []

        # 각 데이터에 대해 분할된 segment와 label을 쌍으로 저장
        for item in preprocessed_dataset:
            ecg_data = item['data']  # ECG 데이터
            label = item['label']  # 레이블 (예: 정상, 비정상 등)
            segmented_ecg = segment_ecg(ecg_data)  # ECG 데이터를 분할

            # 분할된 segment와 label을 쌍으로 저장
            for segment in segmented_ecg:
                self.data_label_pairs.append((segment, label))

    def __len__(self):
        """데이터셋의 전체 길이를 반환"""
        return len(self.data_label_pairs)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 (segment, label) 쌍을 반환. PyTorch Dataloader가 이 메서드를 호출함.
        데이터를 텐서로 변환하여 반환.
        """
        ecg_segment, label = self.data_label_pairs[idx]
        ecg_tensor = torch.tensor(ecg_segment, dtype=torch.float32)
        if label == 'N':
            label = 0
        elif label == 'A':
            label = 1
        elif label == 'O':
            label = 2
        elif label == '~':
            label = 3

        label_tensor = torch.tensor(label, dtype=torch.long)  # 레이블은 보통 정수형
        return ecg_tensor, label_tensor
