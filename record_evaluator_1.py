import os

import numpy as np

from i_signal_reader import SignalReader


class RecordEvaluator:
    def __init__(self, dest_dir):
        self._dest_dir = dest_dir

    def evaluate(self, signal_reader: SignalReader):
        signal = signal_reader.read_signal()
        fs = signal_reader.read_fs()

        result = np.round(np.random.random(size=[signal.shape[0], ]))

        code = signal_reader.get_code()
        np.save(os.path.join(self._dest_dir, f'{code}'), result)
