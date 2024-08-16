import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os

class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings_file, labels_file, seq_length=192, pred_length=24, mode="train", missing_ratio=0.1, seed=0):
        np.random.seed(seed)
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.mode = mode

        path = f"./data/embeddings_missing{missing_ratio}_seed{seed}.pk"

        if not os.path.isfile(path):
            # Load and process data
            self.embeddings = np.load(embeddings_file)
            self.labels = np.load(labels_file)
            
            # Normalize embeddings
            self.embeddings, self.mean_data, self.std_data = self.normalize_data(self.embeddings)
            
            # Create masks
            self.observed_masks = np.ones_like(self.embeddings)
            self.gt_masks = self.create_gt_masks(self.observed_masks, missing_ratio)
            
            with open(path, "wb") as f:
                pickle.dump([self.embeddings, self.observed_masks, self.gt_masks, self.labels, self.mean_data, self.std_data], f)
        else:
            with open(path, "rb") as f:
                self.embeddings, self.observed_masks, self.gt_masks, self.labels, self.mean_data, self.std_data = pickle.load(f)

        self.set_mode(mode)

    def normalize_data(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / std, mean, std

    def create_gt_masks(self, observed_masks, missing_ratio):
        gt_masks = observed_masks.copy()
        obs_indices = np.where(gt_masks.reshape(-1))[0]
        miss_indices = np.random.choice(obs_indices, int(len(obs_indices) * missing_ratio), replace=False)
        gt_masks.reshape(-1)[miss_indices] = 0
        return gt_masks

    def set_mode(self, mode):
        self.mode = mode
        total_length = len(self.embeddings)
        if mode == 'train':
            start = 0
            end = int(total_length * 0.7)
        elif mode == 'valid':
            start = int(total_length * 0.7)
            end = int(total_length * 0.85)
        else:  # test
            start = int(total_length * 0.85)
            end = total_length

        self.use_indices = np.arange(start, end - self.seq_length + 1)

    def __len__(self):
        return len(self.use_indices)

    def __getitem__(self, org_index):
        index = self.use_indices[org_index]
        s = {
            "observed_data": self.embeddings[index:index+self.seq_length],
            "observed_mask": self.observed_masks[index:index+self.seq_length],
            "gt_mask": self.gt_masks[index:index+self.seq_length],
            "timepoints": np.arange(self.seq_length),
            "labels": self.labels[index:index+self.seq_length],
        }
        
        # Set the last pred_length timesteps of gt_mask to 0 for forecasting task
        s["gt_mask"][-self.pred_length:] = 0

        return s

def get_dataloader(embeddings_file, labels_file, batch_size=16, missing_ratio=0.1, seed=1):
    dataset = EmbeddingsDataset(embeddings_file, labels_file, missing_ratio=missing_ratio, seed=seed)
    
    dataset.set_mode("train")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    dataset.set_mode("valid")
    valid_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    dataset.set_mode("test")
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    scaler = dataset.std_data
    mean_scaler = dataset.mean_data

    return train_loader, valid_loader, test_loader, scaler, mean_scaler
