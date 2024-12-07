from utils.datasets import MIT_BIH_Arythmia_Long, MIT_BIH_Arythmia
from models.SimpleConv import SimpleConv
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np

def create_sampler(targets):
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(targets), replacement=True)
    return sampler

def get_target_vector(dataset):
    targets = [target for _, target in dataset]
    return np.array(targets)

def main():
    ds = MIT_BIH_Arythmia_Long(100,5,fs=100,dataset_dir='Datasets/temp/physionet.org/files/ltafdb/1.0.0/')
    # ds = MIT_BIH_Arythmia(100,5,fs=100)
    train_set, val_set = random_split(ds, [0.8, 0.2])
    print("train samplet start")
    targets = get_target_vector(train_set)
    print("mm")
    temp = create_sampler(targets)
    print("train samplet done")
    train = DataLoader(train_set, sampler=temp, batch_size=32, num_workers=8)
    targets_val = get_target_vector(val_set)
    temp_val = create_sampler(targets_val)
    val = DataLoader(val_set, sampler=temp_val, batch_size=32, num_workers=8)
    model = SimpleConv()
    model.train_model(train,val,num_epochs=90)


if __name__ == '__main__':
    main()