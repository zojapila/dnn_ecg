import os
import re
import numpy as np
import wfdb
import pickle
import scipy
from torch.utils.data import Dataset
import torch
import json

def extract_segment_with_padding(z, k, N):
    start_idx = k - N
    end_idx = k + N + 1
    if start_idx < 0:
        padding_left = np.median(z[:end_idx])
        segment = np.concatenate([np.full(-start_idx, padding_left), z[:end_idx]])
    elif end_idx > len(z):
        padding_right = np.median(z[start_idx:])
        segment = np.concatenate([z[start_idx:], np.full(end_idx - len(z), padding_right)])
    else:
        segment = z[start_idx:end_idx]
    return segment


class MIT_BIH_Arythmia(Dataset):
    def __init__(self,N, M, dataset_dir = 'Datasets/files/', fs = 10, filename = "MIT-BIH_Arrythmia.json"):
        """
        n - number of samples of orginal signal resampled to fs, interval [-n,n]
        m - qrs times, interval [-m,m]
        """
        ecg_list = []
        exclusion_lst = ["00735", "03665", "04043", "04936", "05091", "06453", "08378", "08405", "08434", "08455"]
        for file in os.listdir(dataset_dir):
            name = re.match(r'^(.*\d\d+)\.atr$', file)
            if name:
                if name.group(1) in exclusion_lst:
                    continue
            if name:
                record = wfdb.rdsamp(f"{dataset_dir}{name.group(1)}") 
                annotation = wfdb.rdann(f"{dataset_dir}{name.group(1)}", 'atr')
                signal = record[0][:,0]
                fs_original = record[1]["fs"]
                num_samples_target = int(signal.shape[0] * fs / fs_original)
                resampled_signal = scipy.signal.resample(signal, num_samples_target)
                annotation_times_resampled = (annotation.sample * fs) / fs_original
                resampled_annotation = wfdb.Annotation('atr',annotation.symbol,annotation_times_resampled.astype(int),aux_note=annotation.aux_note)
                ecg_list.append({"name": name.group(1),"rec" : resampled_signal, "ann" : resampled_annotation})
        self.samples_list = []
        self.label_list = []
        self.qrs_samples = []
        for dic in ecg_list:
            print(dic["name"])
            # xqrs = wfdb.processing.XQRS(sig=dic["rec"], fs=fs)
            # xqrs.detect()
            # qrs_inds = xqrs.qrs_inds
            for n,i in enumerate(dic["ann"].sample):
                self.label_list.append(1 if dic["ann"].aux_note[n] == '(AFIB' else 0)
                self.samples_list.append(list(extract_segment_with_padding(dic["rec"], dic["ann"].sample[n],N)))
                # nearest_qrs_idx = find_nearest_qrs_index(dic["ann"].sample[n], qrs_inds)
                # self.qrs_samples.append(list(extract_segment_with_padding(qrs_inds,nearest_qrs_idx,M)))
        data = {
            'samples_list': self.samples_list,  # This would work if the segments are simple numeric lists
            'label_list': self.label_list,
            'qrs_samples': self.qrs_samples
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
                
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        data = torch.Tensor(self.samples_list[idx]).unsqueeze(0)
        label = self.label_list[idx]
        return data, label


class MIT_BIH_Arythmia_Base(Dataset):
    def __init__(self, N, M, dataset_dir='Datasets/files/', fs=10, output_dir="processed_data3/", histogram_path=None, data_split=True):
        self.N = N
        self.M = M
        self.dataset_dir = dataset_dir
        self.fs = fs
        self.output_dir = output_dir
        self.cumulative_histogram = []
        self.afibs = 0
        self.histogram_path = histogram_path
        self.data_split = data_split
        self.process_records()

    def process_records(self):
        """Przetwarzanie rekordów."""
        if self.histogram_path and os.path.exists(self.histogram_path):
            with open(self.histogram_path, 'rb') as f:
                self.cumulative_histogram = pickle.load(f)
            print("Załadowano histogram:", self.histogram_path)
        else:
            os.makedirs(self.output_dir, exist_ok=True)
            afib_dir = os.path.join(self.output_dir, 'af/')
            normal_dir = os.path.join(self.output_dir, 'normal/')
            os.makedirs(afib_dir, exist_ok=True)
            os.makedirs(normal_dir, exist_ok=True)
            exclusion_lst = ["00735", "03665", "04043", "04936", "05091", "06453", "08378", "08405", "08434", "08455"]
            start_idx = 0

            for file in os.listdir(self.dataset_dir):
                name = re.match(r'^(.*\d\d+)\.atr$', file)
                if name and name.group(1) not in exclusion_lst:
                    print(f"Przetwarzanie: {name.group(1)}")
                    record = wfdb.rdsamp(f"{self.dataset_dir}{name.group(1)}")
                    annotation = wfdb.rdann(f"{self.dataset_dir}{name.group(1)}", 'atr')
                    signal = record[0][:, 0]
                    fs_original = record[1]["fs"]
                    num_samples_target = int(signal.shape[0] * self.fs / fs_original)
                    resampled_signal = scipy.signal.resample(signal, num_samples_target)
                    annotation_times_resampled = (annotation.sample * self.fs) / fs_original
                    if self.data_split:
                        afib_segments = []
                        normal_segments = []

                        for i, aux_note in enumerate(annotation.aux_note):
                            sample_idx = int(annotation_times_resampled[i])
                            segment = extract_segment_with_padding(resampled_signal, sample_idx, self.N)

                            if "(AFIB" in aux_note:
                                afib_segments.append(segment.tolist())
                                self.afibs += 1
                            else:
                                normal_segments.append(segment.tolist())

                        # afib
                        if afib_segments:
                            afib_file = os.path.join(afib_dir, f"{name.group(1)}_afib.pkl")
                            with open(afib_file, 'wb') as f:
                                pickle.dump(afib_segments, f)

                        # normal
                        if normal_segments:
                            normal_file = os.path.join(normal_dir, f"{name.group(1)}_normal.pkl")
                            with open(normal_file, 'wb') as f:
                                pickle.dump(normal_segments, f)
                    
                    else:
                        all_segments = []

                        for i, aux_note in enumerate(annotation.aux_note):
                            sample_idx = int(annotation_times_resampled[i])
                            segment = extract_segment_with_padding(resampled_signal, sample_idx, self.N)
                            all_segments.append(segment.tolist())
                        
                        if all_segments:
                            all_file = os.path.join(normal_dir, f"{name.group(1)}.pkl")
                            with open(all_file, 'wb') as f:
                                pickle.dump(all_segments, f)

                    num_samples = len(annotation.sample)
                    self.cumulative_histogram.append((start_idx, start_idx + num_samples, name.group(1)))
                    start_idx += num_samples

            histogram_path = os.path.join(self.output_dir, "cumulative_histogram.pkl")
            with open(histogram_path, 'wb') as f:
                pickle.dump(self.cumulative_histogram, f)
            print("Przetwarzanie zakończone. Dane zapisane w:", self.output_dir)

    def __len__(self):
        # return self.cumulative_histogram[-1][1]
        return int(self.afibs * 1.5)

    def __getitem__(self, idx):
        if idx < self.afibs:
            pass
        else:
            pass

    def count_afibs(self):
        """Liczba segmentów AFIB."""
        afib_count = 0
        for start, end, filename in self.cumulative_histogram:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            for aux_note in data["ann"]["aux_note"]:
                if "(AFIB" in aux_note:
                    afib_count += 1
        return afib_count


class MIT_BIH_Arythmia_Long(Dataset):
    def __init__(self, N, M, dataset_dir='Datasets/files/', fs=10, output_dir="processed_data/", histogram_path=None):
        self.N = N
        self.M = M
        if histogram_path and os.path.exists(histogram_path):
            with open(histogram_path, 'rb') as f:
                self.cumulative_histogram = pickle.load(f)
            print("Załadowano histogram:", histogram_path)
        else:
            os.makedirs(output_dir, exist_ok=True)
            exclusion_lst = ["00735", "03665", "04043", "04936", "05091", "06453", "08378", "08405", "08434", "08455"]
            self.cumulative_histogram = []
            start_idx = 0
            for file in os.listdir(dataset_dir):
                name = re.match(r'^(.*\d\d+)\.atr$', file)
                if name and name.group(1) not in exclusion_lst:
                    print(f"Przetwarzanie: {name.group(1)}")
                    record = wfdb.rdsamp(f"{dataset_dir}{name.group(1)}")
                    annotation = wfdb.rdann(f"{dataset_dir}{name.group(1)}", 'atr')
                    signal = record[0][:, 0]
                    fs_original = record[1]["fs"]
                    num_samples_target = int(signal.shape[0] * fs / fs_original)
                    resampled_signal = scipy.signal.resample(signal, num_samples_target)
                    annotation_times_resampled = (annotation.sample * fs) / fs_original
                    data = {
                        "rec": resampled_signal,
                        "ann": {
                            "sample": annotation_times_resampled.astype(int).tolist(),
                            "aux_note": annotation.aux_note
                        }
                    }
                    output_filename = os.path.join(output_dir, f"{name.group(1)}.pkl")
                    with open(output_filename, 'wb') as f:
                        pickle.dump(data, f)
                    num_samples = len(data["ann"]["sample"])
                    self.cumulative_histogram.append((start_idx, start_idx + num_samples, output_filename))
                    start_idx += num_samples
            histogram_path = os.path.join(output_dir, "cumulative_histogram.pkl")
            with open(histogram_path, 'wb') as f:
                pickle.dump(self.cumulative_histogram, f)
            print("Przetwarzanie zakończone. Dane zapisane w:", output_dir)

    def __len__(self):
        return self.cumulative_histogram[-1][1]

    def __getitem__(self, idx):
    # Przejdź przez histogram skumulowany, aby znaleźć odpowiedni plik i zakres indeksów
        for start, end, filename in self.cumulative_histogram:
            if start <= idx < end:
                local_idx = idx - start  # Oblicz indeks lokalny w danym pliku
                break
        else:
            raise IndexError("Index out of range")  # Jeśli nie znajdziesz odpowiedniego zakresu, zgłoś błąd
        
        # Załaduj dane z odpowiedniego pliku
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Pobierz sygnał EKG i informacje o annotacjach
        rec = data["rec"]
        sample_idx = data["ann"]["sample"][local_idx]  # Indeks próbki w danym pliku
        aux_note = data["ann"]["aux_note"][local_idx]  # Etykieta (np. AFIB lub NORMAL)
        
        # Wyciąć odpowiedni segment EKG wokół punktu annotacji
        segment = extract_segment_with_padding(rec, sample_idx, self.N)
        
        # Ustal etykietę: 1 dla AFIB, 0 dla NORMAL
        label = 1 if aux_note == '(AFIB' else 0
        return torch.Tensor(segment).unsqueeze(0), label

    def count_afibs(self):
        afib_count = 0
        for start, end, filename in self.cumulative_histogram:
            # Załaduj dane z pliku
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            # Sprawdź wszystkie etykiety i policz AFIB
            for aux_note in data["ann"]["aux_note"]:
                if "(AFIB" in aux_note:
                    afib_count += 1
        return afib_count