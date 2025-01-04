import os

import numpy as np
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
import wfdb.processing
# from signal_reader_original import SignalReader
from signal_reader import SignalReader
from scipy.signal import resample


import torch
import torch.nn as nn

MODEL_PATH = "test_1_small_but_long.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resample_signal(signal, orig_fs, target_fs):
    num_samples = int(len(signal) * target_fs / orig_fs)
    resampled_signal = resample(signal, num_samples)
    return resampled_signal

class SimpleConv(nn.Module): #definicja modelu potrzebna do ewaluaci przez to że mamy tylko wagi
    def __init__(self, input = 201, input_ch = 1, num_classes = 2): 
        super(SimpleConv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_ch, 64, kernel_size=7, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding='same'),  # out 1 x 128 x n
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # out 1 x 128 x n//2
            nn.Conv1d(128, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding='same'),  # out 1 x 256 x n//2
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),  # out 1 x 256 x n//4
            nn.Conv1d(256, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding='same'),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),  # out 1 x 512 x n//8
            nn.Flatten(),
            nn.Linear(512 * (input // 8), 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)
    

class RecordEvaluator:
    def __init__(self, dest_dir):
        self._dest_dir = dest_dir
        self._model = SimpleConv()
        self._model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        self._model = self._model.to(device)
        self._model.eval()

    def evaluate(self, signal_reader):
        # Odczytanie sygnału i częstotliwości próbkowania
        signal = signal_reader.read_signal()
        fs = signal_reader.read_fs()

        # Detekcja punktów QRS
        xqrs = wfdb.processing.XQRS(sig=signal[:, 0], fs=fs)
        xqrs.detect()
        qrs_inds = xqrs.qrs_inds

        # Obliczanie odstępów RR
        rr = wfdb.processing.calc_rr(qrs_inds, fs=fs, min_rr=None, max_rr=None, qrs_units='samples',
                                     rr_units='seconds')

        # Parametry okna i wsadu
        input_rr_samples = 30
        batch_size = 64
        qrs_af_probabs = np.zeros(shape=(len(qrs_inds),), dtype=np.float32)
        qrs_af_overlap = np.zeros(shape=(len(qrs_inds),), dtype=np.float32)
        pred_step = input_rr_samples // 3

        # Przygotowanie danych
        batch = torch.zeros((batch_size, 1, input_rr_samples), dtype=torch.float32, device=device)
        batch_idx = 0
        rr_indices_history = []

        # with torch.no_grad():  # Wyłączenie obliczeń gradientów
        #     for rr_idx in range(0, rr.shape[0] - input_rr_samples, pred_step):
        #         snippet = rr[rr_idx:rr_idx + input_rr_samples]
        #         rr_indices_history.append([rr_idx, rr_idx + input_rr_samples])
        #         batch[batch_idx, 0, :] = torch.tensor(snippet, dtype=torch.float32).to(self._device)
        #         batch_idx += 1

        #         # Przetwarzanie pełnego wsadu
        #         if batch_idx == batch_size:
        #             results = self._model(batch).cpu().numpy()  # Wyniki na CPU
        #             for j in range(batch_idx):
        #                 rr_from, rr_to = rr_indices_history[j]
        #                 qrs_af_probabs[rr_from: rr_to] += results[j, :, 0]
        #                 qrs_af_overlap[rr_from: rr_to] += 1.0

        #             batch_idx = 0
        #             rr_indices_history = []

        #     # Przetwarzanie pozostałych danych
        #     if batch_idx > 0:
        #         results = self._model(batch[:batch_idx]).cpu().numpy()  # Wyniki na CPU
        #         for j in range(batch_idx):
        #             rr_from, rr_to = rr_indices_history[j]
        #             qrs_af_probabs[rr_from: rr_to] += results[j, :, 0]
        #             qrs_af_overlap[rr_from: rr_to] += 1.0
        with torch.no_grad():
            for rr_idx in range(0, rr.shape[0] - input_rr_samples, pred_step):
                snippet = rr[rr_idx:rr_idx + input_rr_samples]
                rr_indices_history.append([rr_idx, rr_idx + input_rr_samples])
                
                # Konwertujemy snippet na tensor Pytorch
                batch[batch_idx, 0, :] = torch.tensor(snippet, dtype=torch.float32).to(device)
                batch_idx += 1

                if batch_idx == batch_size:
                    # Wyniki modelu - upewnij się, że dane wejściowe to tensory
                    results = self._model(batch).cpu().numpy()
                    for j in range(batch_idx):
                        rr_from, rr_to = rr_indices_history[j]
                        qrs_af_probabs[rr_from: rr_to] += results[j, :, 0]
                        qrs_af_overlap[rr_from: rr_to] += 1.0

                    batch_idx = 0
                    rr_indices_history = []

            if batch_idx > 0:
                # Obsługa ostatniego batcha (częściowego)
                results = self._model(batch[:batch_idx]).cpu().numpy()
                for j in range(batch_idx):
                    rr_from, rr_to = rr_indices_history[j]
                    qrs_af_probabs[rr_from: rr_to] += results[j, :, 0]
                    qrs_af_overlap[rr_from: rr_to] += 1.0


        # Normalizacja wyników
        qrs_af_overlap[qrs_af_overlap == 0.0] = 1.0
        qrs_af_probabs /= qrs_af_overlap
        qrs_af_preds = np.round(qrs_af_probabs)

        # Przypisanie predykcji do całego sygnału
        pred = np.zeros([len(signal), ], dtype=np.float32)
        for qrs_idx in range(len(rr)):
            pred[qrs_inds[qrs_idx]: qrs_inds[qrs_idx + 1]] = qrs_af_preds[qrs_idx]

        # Zapis wyników
        code = signal_reader.get_code()
        np.save(os.path.join(self._dest_dir, f'{code}'), pred)


    
    # def evaluate(self, signal_reader: SignalReader):
    #     signal = signal_reader.read_signal()
    #     fs = signal_reader.read_fs()
    #     target_fs = 100  # Docelowa częstotliwość próbkowania (zgodna z modelem)
    #     if fs != target_fs:
    #         signal = np.apply_along_axis(lambda x: resample_signal(x, fs, target_fs), axis=0, arr=signal)
    #         fs = target_fs 

    #     xqrs = wfdb.processing.XQRS(sig=signal[:, 0], fs=fs)
    #     xqrs.detect()
    #     qrs_inds = xqrs.qrs_inds
    #     rr = wfdb.processing.calc_rr(qrs_inds, fs=fs, min_rr=None, max_rr=None, qrs_units='samples',
    #                                  rr_units='seconds')

    #     input_rr_samples = 30
    #     batch_size = 64
    #     qrs_af_probabs = np.zeros(shape=(len(qrs_inds), ), dtype=np.float32)
    #     qrs_af_overlap = np.zeros(shape=(len(qrs_inds), ), dtype=np.float32)

    #     pred_step = input_rr_samples // 3

    #     batch = np.zeros(shape=(batch_size, input_rr_samples, 1), dtype=np.float32)
    #     batch_idx = 0
    #     rr_indices_history = []
    #     for rr_idx in range(0, rr.shape[0]-input_rr_samples, pred_step):
    #         snippet = rr[rr_idx:rr_idx + input_rr_samples]
    #         rr_indices_history.append([rr_idx, rr_idx + input_rr_samples])
    #         snippet = snippet[..., np.newaxis]
    #         batch[batch_idx] = snippet
    #         batch_idx += 1

    #         if batch_idx == batch_size:
    #             results = self._model(batch)
    #             for j in range(batch_idx):
    #                 rr_from, rr_to = rr_indices_history[j]
    #                 qrs_af_probabs[rr_from: rr_to] += results[j, :, 0]
    #                 qrs_af_overlap[rr_from: rr_to] += 1.0

    #             batch_idx = 0
    #             rr_indices_history = []

    #     if batch_idx > 0:
    #         results = self._model(batch)
    #         for j in range(batch_idx):
    #             rr_from, rr_to = rr_indices_history[j]
    #             qrs_af_probabs[rr_from: rr_to] += results[j, :, 0]
    #             qrs_af_overlap[rr_from: rr_to] += 1.0

    #     qrs_af_overlap[qrs_af_overlap == 0.0] = 1.0
    #     qrs_af_probabs /= qrs_af_overlap
    #     qrs_af_preds = np.round(qrs_af_probabs)

    #     pred = np.zeros([len(signal), ], dtype=np.float32)

    #     for qrs_idx in range(len(rr)):
    #         pred[qrs_inds[qrs_idx]: qrs_inds[qrs_idx+1]] = qrs_af_preds[qrs_idx]

    #     code = signal_reader.get_code()
    #     np.save(os.path.join(self._dest_dir, f'{code}'), pred)


if __name__ == '__main__':
    record_eval = RecordEvaluator('./')
    signal_reader = SignalReader('./val_db/6.csv')
    record_eval.evaluate(signal_reader)
