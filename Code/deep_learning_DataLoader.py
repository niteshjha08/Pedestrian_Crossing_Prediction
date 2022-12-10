import torch
import torch.utils.data as dt

import numpy as np


class CrossingDataset(dt.Dataset):
    def __init__(self, data_points):
        super().__init__()
        self.data_points = data_points

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        data_frame = self.data_points[idx]
        sample = {
            "video_id": data_frame[0],
            "ped_id": data_frame[1],
            "frame_num": data_frame[2],
            "features": torch.from_numpy(np.array(data_frame[4:], dtype=np.float32)),
            "output": torch.from_numpy(np.array(data_frame[3], dtype=np.float32))}
        return sample