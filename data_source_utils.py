import os
import glob

import pandas as pd
import scipy
import numpy as np
import wfdb
import h5py
import ast
import json
from tqdm import tqdm
import re

from dataset_reader import (find_files, load_dataset_1,
                            load_dataset_2, load_dataset_3,
                            load_dataset_4, load_dataset_8,
                            load_generic_dataset)

dataset_info = {
    1: {"name": "1. ECG_high_intensity_exercise", "fs": 250, "path_suffix": "", "load_function": load_dataset_1},
    2: {"name": "2. af-classification", "fs": 300, "path_suffix": "training2017", "load_function": load_dataset_2},
    3: {"name": "3. ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1", "fs": 100, "path_suffix": "records100", "load_function": load_dataset_3},
    4: {"name": "4. ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1", "fs": 500, "path_suffix": "records500", "load_function": load_dataset_4},
    5: {"name": "5. brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0", "fs": None, "path_suffix": "", "load_function": load_generic_dataset},
    6: {"name": "6. a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0", "fs": None, "path_suffix": "", "load_function": load_generic_dataset},
    # MIT-BIH
    8: {"name": "8. ecg-fragment-database-for-the-exploration-of-dangerous-arrhythmia-1.0.0", "fs": None, "path_suffix": "", "load_function": load_dataset_8},
}

def get_files_starting_with_number(directory):

    # 해당 디렉토리 내의 파일 목록 가져오기
    files = os.listdir(directory)

    # 숫자로 시작하는 파일만 필터링
    numbered_files = [f for f in files if re.match(r'^\d+\.', f)]

    return numbered_files

def find_mat_and_hea_files(directory):
    # 하위 디렉토리까지 모두 탐색하여 *.mat 및 *.hea 파일 찾기
    mat_files = []
    hea_files = []
    for root, dirs, files in os.walk(directory):
        mat_files.extend(glob.glob(os.path.join(root, '*.mat')))
        hea_files.extend(glob.glob(os.path.join(root, '*.hea')))

    # 파일 이름에서 확장자를 제거한 후 집합으로 변환
    mat_names = {os.path.splitext(os.path.basename(file))[0] for file in mat_files}
    hea_names = {os.path.splitext(os.path.basename(file))[0] for file in hea_files}

    # mat와 hea가 모두 존재하는 파일 이름 찾기
    common_names = list(mat_names & hea_names)

    return common_names

def process_dataset(n_data, src_dir, offset, labels_dict=None):
    if n_data in dataset_info:
        data_dict = dataset_info[n_data]
        all_contents = data_dict["load_function"](data_dict, src_dir, offset)
        return all_contents
    else:
        print(f"Error: Unknown dataset ID {n_data}")
        return []


def load_all_datasets(src_dir, offset, labels_dict=None):
    all_data_contents = []

    for n_data in dataset_info.keys():  # 모든 데이터셋 번호 반복
        print(f"Processing dataset {n_data}: {dataset_info[n_data]['name']}")
        data_dict = dataset_info[n_data]
        if n_data > 1:
            continue
        dataset_contents  = data_dict["load_function"](data_dict, src_dir, offset,labels_dict)  # 해당 데이터셋 처리

        all_data_contents.append(
            {"data": dataset_contents,
             "fs" : data_dict['fs']}
        )  # 읽어온 데이터를 누적

    return all_data_contents


def load_dataset(offset=3, src_dir="F:\homes\ecg_data", n_data=3):
    '''
    :param src_dir: src destination of dataset
    :param n_data: dataset number
    - 1 : ECG_high_intensity_exercise                                                       250Hz       MAT FILE
            - ecg_segmnets
            - manual_annotations
    - 2 : af-classification                                                                 300Hz       MAT FILE
            - training2017
    - 3 : ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1               100Hz       HEA FILE
            - records100
    - 4 : ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1               500Hz       HEA FILE
            - records500
    - 5 : brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0                              HEA FILE
            - 100001
            - 100002
            - ....
    - 6 : a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0                   MAT FILE
            - 01        - 02
                - 010       - 020
                - 011       - 021
                - ...        - ...
    - 7 : cipa-ecg-validation-study-1.0.0 :일단 패스
    - 8 : ecg-fragment-database-for-the-exploration-of-dangerous-arrhythmia-1.0.0 # 부정맥 분류
    - 9 : non-invasive-fetal-ecg # 부정맥 태아  / 정상리듬 태아
    -10 : paroxysmal-atrial-fibrillation # AF 분류
    -11 : wilson-central-terminal-ecg-database-1.0.1 # segment data
    :return:
    '''

    with open('labels.json', 'r') as f:
        labels_dict = json.load(f)

    dataset_list = get_files_starting_with_number(src_dir)

    data_dict = {}

    for idx, dataset_dir in enumerate(dataset_list):
        dataset_name = dataset_dir.split(' ')[-1]
        print(dataset_name)

    all_contents = load_all_datasets(src_dir, offset, labels_dict)

    return all_contents

if __name__ == "__main__":
    load_dataset()