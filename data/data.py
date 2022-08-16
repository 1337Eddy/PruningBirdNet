from operator import indexOf
from torch.utils.data import Dataset
from utils.audio import get_spec, openAudioFile
import os
import numpy as np


class CallsDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.paths = self.__load_paths__()

    def __len__(self):
        sum = 0
        for _, _, num in self.paths:
            sum += num
        return sum

    def __load_paths__(self):
        paths = []
        birds = os.listdir(self.path)
        birds = sorted(birds)
        for bird in birds:
            calls = os.listdir(self.path + bird + "/")
            calls = filter(lambda x: ".wav" in x, calls)
            calls = sorted(calls)
            paths += [(bird, calls, len(calls))]
        return paths


    def __idx_to_path__(self, idx):
        current = 0
        for bird, calls, num in self.paths:
            if (idx >= current + num):
                current += num
                continue
            else:
                path = self.path + bird + "/" + calls[idx-current]
                return path, bird
        #print(idx)
        raise IndexError("Index" + str(idx) + "not in range")

    def __get_spec_from_path__(self, path, rate=44100):
        sig, rate = openAudioFile(path, rate)
        x = get_spec(sig, rate, shape=(64, 384))
        return x[None, :, :]

    def __getitem__(self, idx):
        path, bird = self.__idx_to_path__(idx)
        return self.__get_spec_from_path__(path), bird
