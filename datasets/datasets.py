import numpy as np
import torch 
from torch.utils.data import Dataset

class SolarData(Dataset):
    """Solar dataset.
    
    This will load the dataset into the memory and is only suitable for small amount of data
    """

    def __init__(self, data1, data2, log = True):
        """
        Args:
            data (np.array): Path to the npz file with annotations.
        """
        super(SolarData, self).__init__()
        self.src = np.concatenate((data1["arr_0"], data2["arr_0"]), axis = 0)
        self.tar = np.concatenate((data1["arr_1"], data2["arr_1"]), axis = 0)

        # Everything smaller than 0 is wrong
        self.src[self.src <= 1.0] = 1.0
        if log:
            self.src = np.log(self.src)
        self.src = self.src.reshape(self.src.shape[0], 3, 256, 256)
        self.tar = self.tar.reshape(self.tar.shape[0], )

        # CenterCrop
        self.src = self.src[:, :, 26:230, 26:230]


    def __len__(self):
        return len(self.tar)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = [self.src[idx-1], self.tar[idx-1]]
        # sample = data_transforms(sample) # convert to tensor

        return sample



# data1 = np.load(base + 'maps_256_6800_flares.npz')
# data2 = np.load(base + 'maps_256_7000_non_flares.npz')

# dataset = SolarData(data1 = data1, data2 = data2)
# train_size = len(dataset) * 4 // 5
# val_size = len(dataset) - train_size

# print(len(dataset), train_size, val_size)
# solar_dataset, valid_dataset = random_split(dataset, [train_size, val_size])