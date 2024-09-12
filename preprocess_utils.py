from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import neurokit2 as nk
from scipy.signal import resample
import json
from datetime import datetime

def resample_signal(signal, original_fs, target_fs):
    """
    신호를 재샘플링하는 함수.

    Parameters:
    - signal: 원본 신호 데이터 (1D numpy array)
    - original_fs: 원본 샘플링 주파수 (Hz)
    - target_fs: 목표 샘플링 주파수 (Hz)

    Returns:
    - resampled_signal: 재샘플링된 신호 데이터 (1D numpy array)
    """
    # 원본 신호 길이
    original_length = len(signal)

    # 재샘플링할 길이 계산 (목표 샘플링 주파수에 따른 길이)
    target_length = int(original_length * target_fs / original_fs)

    # 신호 재샘플링
    resampled_signal = resample(signal, target_length)

    return resampled_signal

def convert_to_image(ecg_data, img_size):
    # ecg_data: (1000, 3) 형태의 데이터
    # img_size: 정사각형 이미지의 크기

    img = np.zeros((img_size, img_size, 3), dtype=np.float32)

    for i in range(ecg_data.shape[0]):
        row = i // img_size
        col = i % img_size
        img[row, col, :] = ecg_data[i, :]

    return img


# IQR 기반 Normalization 함수
def iqr_normalize(data):
    # 1사분위수(Q1)와 3사분위수(Q3) 계산
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # IQR을 이용한 데이터 정규화
    normalized_data = (data - Q1) / IQR
    return normalized_data

def preprocessing(signal, offset=None, fs=250, cutoff=30,plot=True):
    signal = resample_signal(signal,fs,114)
    signal = detrend(signal[:offset], 100)
    signal = min_max_normalize(iqr_normalize(signal))
    if plot:
        plt.plot(signal)
        plt.show()
    return signal

def convert_ndarray_to_list(data):
    """Numpy 배열을 파이썬 리스트로 변환"""
    if isinstance(data, np.ndarray):
        return data.tolist()  # ndarray를 리스트로 변환
    elif isinstance(data, dict):
        # dict의 값이 ndarray일 경우 처리
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    return data

def preprocess_signal(signal, dtype, fs, plot):
    if isinstance(signal, dict):
        signal['data'] = preprocessing(signal['data'], fs=fs, plot=plot)
        signal['data'] = generate_quality_map(signal['data'], start_idx=0, end_idx=None, fs=114, plot=plot)
    else:
        signal = preprocessing(signal, fs=fs, plot=plot)
        signal = generate_quality_map(signal, start_idx=0, end_idx=None, fs=fs, plot=plot)

    if dtype == 2:
        signal['data'] = convert_to_image(signal['data'], int(np.ceil(np.sqrt(signal['data'].shape[0]))))

    return signal

def preprocess_chunk(chunk, dtype, fs, plot, use_parallel, n_jobs):
    if use_parallel:
        preprocessed_chunk = Parallel(n_jobs=n_jobs)(
            delayed(preprocess_signal)(signal, dtype, fs, plot) for signal in chunk
        )
    else:
        preprocessed_chunk = [preprocess_signal(signal, dtype, fs, plot) for signal in chunk]

    # Numpy 배열을 리스트로 변환
    preprocessed_chunk = [convert_ndarray_to_list(signal) for signal in preprocessed_chunk]

    return preprocessed_chunk

def preprocess_dataset(dataset,dtype=1, fs = 100, plot=False, debug=True, use_parallel=True, n_jobs=-1, chunk_size=32):
    preprocessed_data = []

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'F:\homes\preprocessed_data\preprocessed_data_{current_time}'
    chunk_idx = 0
    total_data_len = len(dataset)

    # 데이터 청크 단위로 처리
    for start_idx in tqdm(range(0, total_data_len, chunk_size), desc="Processing dataset"):
        end_idx = min(start_idx + chunk_size, total_data_len)
        chunk = dataset[start_idx:end_idx]

        # 각 청크에 대해 전처리
        preprocessed_chunk = preprocess_chunk(chunk, dtype, fs, plot, use_parallel, n_jobs)

        # 처리된 청크 저장
        if save_path:
            chunk_file = f"{save_path}_chunk_{chunk_idx}.json"
            with open(chunk_file, 'w') as f:
                json.dump(preprocessed_chunk, f, indent=4)
            print(f"Chunk {chunk_idx} has been saved to {chunk_file}")

        chunk_idx += 1

    if len(preprocessed_data) > 0:
        # Find the minimum length among all datasets
        min_length = min([data.shape[0] for data in preprocessed_data])

        # Truncate each dataset to the minimum length
        truncated_data = [data[:min_length] for data in preprocessed_data]

        # Stack the truncated datasets
        stacked_contents = np.stack(truncated_data, axis=0)
        print("Final stacked contents shape:", stacked_contents.shape)

        return stacked_contents
    else:
        print("No valid preprocessed data found.")
        return None


def generate_mapped_signal(signal, fs=300, start_idx=0, end_idx=None, plot=False):

    # 신호의 끝 인덱스를 설정
    if end_idx is None:
        end_idx = len(signal)

    # 신호의 특정 구간을 추출
    signal_segment = signal[start_idx:end_idx]

    # Get R-peaks location
    _, rpeaks = nk.ecg_peaks(signal_segment, sampling_rate=fs)

    if len(rpeaks['ECG_R_Peaks']) < 4:
        return np.zeros(signal_segment.shape)
    # Delineate cardiac cycle
    signals, waves = nk.ecg_delineate(signal_segment, rpeaks, sampling_rate=fs)

    # 초기화: 모든 구간을 0으로 설정
    mapped_signal = np.zeros(len(signal_segment))

    # P 파형 영역 매핑
    for onset, offset in zip(waves["ECG_P_Onsets"], waves["ECG_P_Offsets"]):
        if not np.isnan(onset) and not np.isnan(offset):
            mapped_signal[int(onset):int(offset) + 1] = 1

    # QRS 복합파 영역 매핑
    for onset, offset in zip(waves["ECG_R_Onsets"], waves["ECG_R_Offsets"]):
        if not np.isnan(onset) and not np.isnan(offset):
            mapped_signal[int(onset):int(offset) + 1] = 2

    # T 파형 영역 매핑
    for onset, offset in zip(waves["ECG_T_Onsets"], waves["ECG_T_Offsets"]):
        if not np.isnan(onset) and not np.isnan(offset):
            mapped_signal[int(onset):int(offset) + 1] = 3

    if plot:
        # 파형 영역 시각화
        plt.figure(figsize=(15, 6))
        plt.plot(signal_segment, label="ECG Signal")

        # P 파형 영역 시각화
        for onset, offset in zip(waves["ECG_P_Onsets"], waves["ECG_P_Offsets"]):
            if not np.isnan(onset) and not np.isnan(offset):
                plt.axvspan(onset, offset, color='yellow', alpha=0.3,
                            label='P-wave' if 'P-wave' not in plt.gca().get_legend_handles_labels()[1] else "")

        # QRS 복합파 영역 시각화 (Q, R, S 파형 포함)
        for q_onset, r_peak, s_offset in zip(waves["ECG_R_Onsets"], rpeaks['ECG_R_Peaks'], waves["ECG_R_Offsets"]):
            if not np.isnan(q_onset) and not np.isnan(s_offset):
                plt.axvspan(q_onset, s_offset, color='red', alpha=0.3,
                            label='QRS complex' if 'QRS complex' not in plt.gca().get_legend_handles_labels()[
                                1] else "")
                plt.axvline(r_peak, color='black', linestyle='--',
                            label='R-peak' if 'R-peak' not in plt.gca().get_legend_handles_labels()[1] else "")

        # T 파형 영역 시각화
        for onset, offset in zip(waves["ECG_T_Onsets"], waves["ECG_T_Offsets"]):
            if not np.isnan(onset) and not np.isnan(offset):
                plt.axvspan(onset, offset, color='blue', alpha=0.3,
                            label='T-wave' if 'T-wave' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.legend()
        plt.title(f"ECG Signal with P, QRS, T wave regions (Samples {start_idx} to {end_idx})")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.show()

    return mapped_signal


def generate_quality_map(signal, fs=300, start_idx=0, end_idx=None, plot=False):
    # 신호의 끝 인덱스를 설정
    if end_idx is None:
        end_idx = len(signal)

    # 신호의 특정 구간을 추출
    signal_segment = signal[start_idx:end_idx]


    # 품질 지수 계산
    try:
        quality = nk.ecg_quality(signal_segment, sampling_rate=fs)
    except Exception as e:
        print(f"Error in nk.ecg_quality: {e}. Filling quality with zeros.")
        quality = np.zeros(signal_segment.shape)
    # 품질 지수를 0과 1 사이로 클리핑
    quality = np.clip(quality, 0, 1)

    # mapped_signal 생성
    mapped_signal = generate_mapped_signal(signal_segment, fs, 0, None, plot)

    # 스케일링
    signal_scaled = (signal_segment * 255).astype(np.uint8)
    quality_scaled = (quality * 255).astype(np.uint8)

    # mapped_signal 값을 64, 128, 192, 255로 매핑
    mapped_signal = mapped_signal.astype(np.uint8)
    mapped_signal[mapped_signal == 1] = 64
    mapped_signal[mapped_signal == 2] = 128
    mapped_signal[mapped_signal == 3] = 192
    mapped_signal[mapped_signal == 0] = 0

    # 3채널 numpy array 생성
    combined_array = np.stack((signal_scaled, quality_scaled, mapped_signal), axis=-1)

    if plot:
        # 시각화
        plt.figure(figsize=(10, 10))

        # 2D 이미지로 변환
        img_size = int(np.ceil(np.sqrt(combined_array.shape[0])))
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        for i in range(combined_array.shape[0]):
            row = i // img_size
            col = i % img_size
            img[row, col, :] = combined_array[i, :]

        plt.imshow(img)
        plt.title(f"ECG Signal, Quality Index, and Mapped Signal (Samples {start_idx} to {end_idx})")
        plt.axis('off')
        plt.show()

    return combined_array
#rpeaks['ECG_R_Peaks']