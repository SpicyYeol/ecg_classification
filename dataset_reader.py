import os
import glob
import scipy
import pandas as pd
import tqdm
import json
import numpy as np
import wfdb



def find_files(directory, ext='*.mat'):
    # 하위 디렉토리까지 모두 탐색하여 *.mat 파일 찾기
    mat_files = []
    for root, dirs, files in os.walk(directory):
        for file in glob.glob(os.path.join(root, ext)):
            mat_files.append(file)
    return mat_files
# 데이터 로딩 함수 정의
def load_dataset_1(data_dict, src_dir, offset, labels_dict = None):
    mats = find_files(os.path.join(src_dir, data_dict['name']), '*.mat')
    all_contents = []
    for mat in mats[:offset]:
        mat_data = scipy.io.loadmat(mat)
        if 'ecg_raw' in mat_data:
            mat_contents = mat_data['ecg_raw'][0]
        elif 'ecg' in mat_data:
            mat_contents = mat_data['ecg'][0]
        else:
            print(f"Warning: '{mat}' does not contain 'ecg_raw' or 'ecg'")
            continue
        all_contents.append(mat_contents)
    return all_contents

# load af-classification dataset
def load_dataset_2(data_dict, src_dir, offset, labels_dict = None):
    mats = find_files(os.path.join(src_dir, data_dict['name'], 'training2017'), '*.mat')
    label_csv = os.path.join(src_dir, data_dict['name'], 'REFERENCE-v0.csv')
    label_data = pd.read_csv(label_csv)
    all_contents = []
    for mat in mats[:offset]:
        mat_data = scipy.io.loadmat(mat)
        label = label_data[label_data['filename'] == mat.split('\\')[-1].split('.')[0]].iloc[0]['label']
        if 'val' in mat_data:
            mat_contents = mat_data['val'][0]
        else:
            print(f"Warning: '{mat}' does not contain 'ecg_raw' or 'ecg'")
            continue
        all_contents.append({
            'data': mat_contents,
            'label': label
        })
    return all_contents

def load_dataset_3(data_dict, src_dir, offset, labels_dict):
    ann_path = os.path.join(src_dir, data_dict['name'], 'ptbxl_database.csv')
    df = pd.read_csv(ann_path)
    df_with_scp_codes = df[df['scp_codes'].notna()]
    heas = find_files(os.path.join(src_dir, data_dict['dataset_name'], 'records' + str(data_dict['fs'])), '*.hea')

    all_contents = []
    if offset == -1:
        offset = len(heas)
    for hea in tqdm(heas[:offset], desc='Loading data'):
        record = wfdb.rdrecord(hea[:-4])
        record_name = os.path.basename(hea)[:-4]
        record_metadata = df_with_scp_codes[df_with_scp_codes['filename_lr'].str.contains(record_name)]

        if not record_metadata.empty:
            scp_codes = record_metadata.iloc[0]['scp_codes']
            scp_codes_dict = json.loads(scp_codes.replace("'", "\""))
            one_hot_encoded = np.zeros(len(labels_dict), dtype=float)
            for code, value in scp_codes_dict.items():
                if code in labels_dict:
                    index = labels_dict[code]
                    one_hot_encoded[index] = value / 100.0

            ecg_data = record.p_signal[:, 1]
            all_contents.append({'ecg': ecg_data, 'labels': one_hot_encoded})
    return all_contents

def load_dataset_4(data_dict, src_dir, offset, labels_dict = None):
    heas = find_files(os.path.join(src_dir, data_dict['dataset_name'], 'records' + str(data_dict['fs'])), '*.hea')
    all_contents = []
    for hea in heas[:offset]:
        record = wfdb.rdrecord(hea[:-4])
        all_contents.append(record.p_signal[:,1])
    return all_contents
def load_dataset_8(data_dict, src_dir, offset, labels_dict = None):
    # 여기에 다른 데이터셋에 대한 로직을 추가할 수 있습니다.
    return []
def load_generic_dataset(data_dict, src_dir, offset, labels_dict = None):
    # 여기에 다른 데이터셋에 대한 로직을 추가할 수 있습니다.
    return []
