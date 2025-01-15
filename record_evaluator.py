import os

import numpy as np
import torch
import torch.nn as nn
import wfdb.processing
from signal_reader import SignalReader
import torch.nn.functional as F
from tqdm import tqdm
import keras
import tensorflow as tf
import scipy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConv(nn.Module):
    def __init__(self, input_size=40, input_ch=1, num_classes=2):
        super(SimpleConv, self).__init__()

        # Uproszczona wersja modelu
        self.conv1 = nn.Conv1d(input_ch, 64, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)

        # Obliczymy wymiar po warstwach Conv1d i MaxPool1d
        self.fc1_input_size = 128 * (input_size // 2)  # Po 2 max pooling
        self.fc1 = nn.Linear(self.fc1_input_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Pass through conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Pass through conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Max pooling
        x = self.pool(x)

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def bce_dice_weighted_loss_wrapper(bce_w, dice_w, smooth=10e-6):
    bce_loss = keras.losses.BinaryCrossentropy()
    dice_loss = dice_coef_loss_wrapper(smooth)
    def bce_dice_weighted_loss(y_true, y_pred):
        return bce_w * bce_loss(y_true, y_pred) + dice_w * dice_loss(y_true, y_pred)
    return bce_dice_weighted_loss

def dice_coef_wrapper(smooth=10e-6):
    def dice_coef(y_true, y_pred):
        y_true_f = y_true
        y_pred_f = y_pred
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return dice
    return dice_coef


def dice_coef_loss_wrapper(smooth=10e-6):
    dice_coef = dice_coef_wrapper(smooth)
    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)
    return dice_coef_loss


# class RecordEvaluator:
#     def __init__(self, dest_dir):
#         self._dest_dir = dest_dir
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self._model = SimpleConv().to(device)
#         self._model.load_state_dict(torch.load("simp_conv_qrs.pt", weights_only=True))
#         self._model.eval()
#         self.iteration = 0

#     def evaluate(self, signal_reader: SignalReader):
#         signal = signal_reader.read_signal()[:,0]
#         fs = signal_reader.read_fs()
#         # if self.iteration != 0:
#         #     pred = np.zeros([len(signal), ], dtype=np.float32)
#         #     code = signal_reader.get_code()
#         #     np.save(os.path.join(self._dest_dir, f'{code}'), pred)

#         # self.iteration +=1
#         fs_conv = 100
#         num_samples_target = int(len(signal) * fs_conv / fs)
#         resampled_signal = scipy.signal.resample(signal, num_samples_target)
#         xqrs = wfdb.processing.XQRS(sig=resampled_signal, fs=fs)
#         xqrs.detect()
#         qrs_inds = xqrs.qrs_inds
#         rr = wfdb.processing.calc_rr(qrs_inds, fs=fs, min_rr=None, max_rr=None, qrs_units='samples',
#                                      rr_units='seconds')
#         input_rr_samples = 40
#         batch_size = 64
        
#         starttime = time.time()
#         # cutOff = 20
#         # b, a = scipy.signal.butter(5, cutOff, fs=fs, btype='low', analog=False)
#         # signal = scipy.signal.lfilter(b,a,signal)

#         resampled_labels = np.zeros(len(rr))
#         endtime = time.time()
#         print('preprocessing ', endtime-starttime)
#         starttime = time.time()
#         for idx in range(20,len(resampled_labels)-input_rr_samples, input_rr_samples):
#             input = [rr[idx-20:idx+20]]
#             input = torch.Tensor(input).unsqueeze(0)
#             outputs = self._model.forward(input)
#             _, predicted = torch.max(outputs.data, 1)
#             resampled_labels[idx-20:idx+20] = predicted
#         endtime = time.time()
#         print('evaluation ', endtime-starttime)

#         starttime = time.time()
#         pred = np.zeros([len(signal), ], dtype=np.float32)
#         for i in range(len(pred)):
#             pred[i] = resampled_labels[int(i * fs_conv / fs)]
#         endtime = time.time()
#         print('last resampling ', endtime-starttime)

#         code = signal_reader.get_code()
#         np.save(os.path.join(self._dest_dir, f'{code}'), pred)


# if __name__ == '__main__':
#     record_eval = RecordEvaluator('./')
#     signal_reader = SignalReader('./val_db/6.csv')
#     record_eval.evaluate(signal_reader)
class RecordEvaluator:
    def __init__(self, dest_dir):
        self._dest_dir = dest_dir
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Wybierz urządzenie
        self.device = device
        self._model = SimpleConv().to(device)  # Przenieś model na urządzenie
        self._model.load_state_dict(torch.load("simp_conv_qrs.pt", weights_only=True, map_location=device))
        self._model.eval()

    def evaluate(self, signal_reader: SignalReader):
        signal = signal_reader.read_signal()[:,0]
        fs = signal_reader.read_fs()

        fs_conv = 100
        num_samples_target = int(len(signal) * fs_conv / fs)
        resampled_signal = scipy.signal.resample(signal, num_samples_target)
        xqrs = wfdb.processing.XQRS(sig=resampled_signal, fs=fs)
        xqrs.detect()
        qrs_inds = xqrs.qrs_inds
        rr = wfdb.processing.calc_rr(qrs_inds, fs=fs, min_rr=None, max_rr=None, qrs_units='samples',
                                     rr_units='seconds')
        input_rr_samples = 40
        batch_size = 64

        resampled_labels = np.zeros(len(rr))

        for idx in range(20, len(resampled_labels) - 20):
            input = [rr[idx-20:idx+20]]
            input = torch.Tensor(input).unsqueeze(0).to(self.device)  # Przenieś dane na odpowiednie urządzenie
            outputs = self._model.forward(input)
            _, predicted = torch.max(outputs.data, 1)
            resampled_labels[idx] = predicted.cpu().numpy()  # Przenieś wyniki na CPU

        pred = np.zeros([len(signal), ], dtype=np.float32)
        # for i in range(len(pred)):
        #     pred[i] = resampled_labels[int(i * fs_conv / fs)]
        # endtime = time.time()
        for i in range(len(pred)):
            index = int(i * fs_conv / fs)
            if index < len(resampled_labels):  # Sprawdź, czy indeks mieści się w granicach
                pred[i] = int(resampled_labels[index])
            else:
                pred[i] = 0  # Możesz ustawić domyślną wartość, np. 0


        code = signal_reader.get_code()
        np.save(os.path.join(self._dest_dir, f'{code}'), pred)
