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

class ResNetBlock(nn.Module):
    def __init__(self,in_channels, out_channels):
        """
        output same as input
        """
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace=False))  # Changed inplace to False
        self.conv2 = nn.Sequential(
                        nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace=False))
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        if(in_channels != out_channels):
            self.residual = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.in_channels != self.out_channels:
            residual = self.residual(x)
        else:
            residual = x
        return F.relu(out + residual, inplace=False)


class ResNetLike(nn.Module):
    def __init__(self, input = 201, input_ch = 1, num_classes = 2):
        super(ResNetLike, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_ch, 64, kernel_size=7, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResNetBlock(64,64),
            ResNetBlock(64,64),
            ResNetBlock(64,64),
            ResNetBlock(64,128),    # out 1 x 128 x n
            nn.MaxPool1d(2),        # out 1 x 128 x n//2
            ResNetBlock(128,128),
            ResNetBlock(128,128),
            ResNetBlock(128,256),
            nn.MaxPool1d(2),        # out 1 x 256 x n//2
            ResNetBlock(256,256),
            ResNetBlock(256,256),
            ResNetBlock(256,512),
            nn.MaxPool1d(2),        # out 1 x 512 x n//8
            nn.Flatten(),
            nn.Linear(512*(input//8), 256),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):

        return self.model(x)
    
    def train_model(self, train_loader, valid_loader, num_epochs = 5, learning_rate=0.001, save_best = False, save_thr = 0.94):
        best_accuracy = 0.0
        total_step = len(train_loader)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

        for epoch in range(num_epochs):
            # self.train()
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(tqdm(train_loader)):
                # Move tensors to the configured device
                images = images.float()
                labels = labels.type(torch.LongTensor)
                labels = labels


                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                # Backward and optimize
                loss.backward()
                
                optimizer.step()

                # accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct += (torch.eq(predicted, labels)).sum().item()
                total += labels.size(0)

                del images, labels, outputs

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'
                            .format(epoch+1, num_epochs, i+1, total_step, loss.item(), (float(correct))/total))


            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Validation
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in valid_loader:
                    images = images.float()
                    labels = labels
                    outputs = self.forward(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (torch.eq(predicted, labels)).sum().item()
                    del images, labels, outputs
                if(((100 * correct / total) > best_accuracy) and save_best and ((100 * correct / total) > save_thr)):
                    torch.save(self.state_dict(), "best_resnet50_MINST-DVS2.pt")

                print('Accuracy of the network: {} %'.format( 100 * correct / total))



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


class RecordEvaluator:
    def __init__(self, dest_dir):
        self._dest_dir = dest_dir
        self._model = ResNetLike()
        self._model.load_state_dict(torch.load("best_resnet50_afdb_filtration_20Hz_cpu.pt", weights_only=True))
        self._model.eval()
        self.iteration = 0

    # def evaluate(self, signal_reader: SignalReader):
    #     signal = signal_reader.read_signal()[:,0]
    #     fs = signal_reader.read_fs()
    #     # if self.iteration != 0:
    #     #     pred = np.zeros([len(signal), ], dtype=np.float32)
    #     #     code = signal_reader.get_code()
    #     #     np.save(os.path.join(self._dest_dir, f'{code}'), pred)

    #     self.iteration +=1
        
    #     starttime = time.time()
    #     cutOff = 20
    #     b, a = scipy.signal.butter(5, cutOff, fs=fs, btype='low', analog=False)
    #     signal = scipy.signal.lfilter(b,a,signal)

    #     fs_conv = 20
    #     num_samples_target = int(len(signal) * fs_conv / fs)
    #     resampled_signal = scipy.signal.resample(signal, num_samples_target)
    #     resampled_labels = np.zeros(len(resampled_signal))
    #     endtime = time.time()
    #     print('preprocessing ', endtime-starttime)
    #     starttime = time.time()
    #     for idx in range(100,len(resampled_labels)-101, 100):
    #         input = [resampled_signal[idx-100:idx+101]]
    #         input = torch.Tensor(input).unsqueeze(0)
    #         outputs = self._model.forward(input)
    #         _, predicted = torch.max(outputs.data, 1)
    #         resampled_labels[idx-50:idx+51] = predicted
    #     endtime = time.time()
    #     print('evaluation ', endtime-starttime)

    #     starttime = time.time()
    #     pred = np.zeros([len(signal), ], dtype=np.float32)
    #     for i in range(len(pred)):
    #         pred[i] = resampled_labels[int(i * fs_conv / fs)]
    #     endtime = time.time()
    #     print('last resampling ', endtime-starttime)

    #     code = signal_reader.get_code()
    #     np.save(os.path.join(self._dest_dir, f'{code}'), pred)
    def evaluate(self, signal_reader: SignalReader):
    # Odczyt sygnału i częstotliwości próbkowania
        signal = signal_reader.read_signal()[:, 0]
        fs = signal_reader.read_fs()

        self.iteration += 1

        # Preprocessing: Detekcja QRS
        starttime = time.time()
        xqrs = wfdb.processing.XQRS(sig=signal, fs=fs)
        xqrs.detect()
        qrs_inds = xqrs.qrs_inds  # Indeksy wykrytych pików QRS
        endtime = time.time()
        print('QRS detection and preprocessing:', endtime - starttime)

        # Przygotowanie danych do modelu
        starttime = time.time()
        resampled_labels = np.zeros(len(signal), dtype=np.float32)

        for idx in range(1, len(qrs_inds) - 1):  # Przetwarzanie między kolejnymi QRS
            # Ekstrakcja fragmentu sygnału wokół QRS
            start = max(0, qrs_inds[idx] - 100)
            end = min(len(signal), qrs_inds[idx] + 101)
            snippet = signal[start:end]

            # Wypełnienie brakujących próbek do wymaganego rozmiaru
            if len(snippet) < 201:
                snippet = np.pad(snippet, (0, 201 - len(snippet)), 'constant')

            # Konwersja na tensor i predykcja
            input = torch.Tensor(snippet).unsqueeze(0).unsqueeze(0)  # [1, 1, 201]
            outputs = self._model.forward(input)
            _, predicted = torch.max(outputs.data, 1)

            # Przypisanie predykcji do zakresu wokół QRS
            resampled_labels[start:end] = predicted.item()

        endtime = time.time()
        print('Model evaluation:', endtime - starttime)

        # Zapis wyników
        code = signal_reader.get_code()
        np.save(os.path.join(self._dest_dir, f'{code}'), resampled_labels)


if __name__ == '__main__':
    record_eval = RecordEvaluator('./')
    signal_reader = SignalReader('./val_db/6.csv')
    record_eval.evaluate(signal_reader)
