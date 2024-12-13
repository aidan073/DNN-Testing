import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class GymDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), self.labels.iloc[idx]
    
def process(dataset_path):
    scaler = StandardScaler()
    data = pd.read_csv(dataset_path)

    # deal with cats
    cats = ["Gender", "Workout_Type"]
    categoricals = data[cats]
    data.drop(cats, axis=1, inplace=True)
    data = pd.concat([data, pd.get_dummies(categoricals)], axis=1)

    # get splits
    labels = data.pop("Experience_Level")
    labels = labels.apply(lambda x: x-1)
    data = scaler.fit_transform(data)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=73)
    test_data, val_data, test_labels, val_labels = train_test_split(test_data, test_labels, test_size=0.5, random_state=73)

    train_data = GymDataset(train_data, train_labels)
    test_data = GymDataset(test_data, test_labels)
    val_data = GymDataset(val_data, val_labels)

    return train_data, test_data, val_data

