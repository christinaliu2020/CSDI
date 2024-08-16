import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os

class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings_file, seq_length=192, pred_length=24, use_index_list=None, missing_ratio=0.1, seed=0):
        self.seq_length = seq_length
        self.pred_length = pred_length
        np.random.seed(seed)

        path = f"./data/embeddings_missing{missing_ratio}_seed{seed}.pk"

        if not os.path.isfile(path):
            self.embeddings = np.load(embeddings_file)
            
            # Normalize embeddings
            self.embeddings, self.mean, self.std = self.normalize_data(self.embeddings)
            
            # Create masks
            self.observed_masks = np.ones_like(self.embeddings)
            self.gt_masks = self.create_gt_masks(self.observed_masks, missing_ratio)
            
            with open(path, "wb") as f:
                pickle.dump([self.embeddings, self.observed_masks, self.gt_masks], f)
        else:
            with open(path, "rb") as f:
                self.embeddings, self.observed_masks, self.gt_masks = pickle.load(f)

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.embeddings) - self.seq_length + 1)
        else:
            self.use_index_list = use_index_list

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

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.embeddings[index:index+self.seq_length],
            "observed_mask": self.observed_masks[index:index+self.seq_length],
            "gt_mask": self.gt_masks[index:index+self.seq_length],
            "timepoints": np.arange(self.seq_length),
        }
        
        # Set the last pred_length timesteps of gt_mask to 0 for forecasting task
        s["gt_mask"][-self.pred_length:] = 0

        return s

    def __len__(self):
        return len(self.use_index_list)

def get_dataloader(embeddings_file, batch_size=16, missing_ratio=0.1, seed=1):
    dataset = EmbeddingsDataset(embeddings_file, missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # Split into train/valid/test
    test_ratio = 0.1
    valid_ratio = 0.2
    test_length = int(len(dataset) * test_ratio)
    valid_length = int(len(dataset) * valid_ratio)

    test_index = indlist[:test_length]
    valid_index = indlist[test_length:test_length+valid_length]
    train_index = indlist[test_length+valid_length:]

    train_dataset = EmbeddingsDataset(embeddings_file, use_index_list=train_index, missing_ratio=missing_ratio, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = EmbeddingsDataset(embeddings_file, use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = EmbeddingsDataset(embeddings_file, use_index_list=test_index, missing_ratio=missing_ratio, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
