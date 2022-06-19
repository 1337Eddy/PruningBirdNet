from operator import indexOf
import time
from torch.utils.data import Dataset
from utils.audio import get_spec, openAudioFile, splitSignal
import os
import numpy as np
import soundfile as sf
path = "1dataset/1data/1calls/"


birds = os.listdir(path)
birds = sorted(birds)

bird_dict = {x: birds.index(x) for x in birds}

def labels_to_one_hot_encondings(labels):
    result = np.zeros((len(labels), len(birds)))
    for i in range(0, len(labels)):
        result[i][bird_dict[labels[i]]] = 1
    return result

def id_to_label(id):
    return list(bird_dict)[id]


class CallsDataset(Dataset):
    def __init__(self, path=path, seconds=3, rate=44100, overlap=0, minlen=3):
        self.path = path
        self.paths = self.__load_paths__()
        self.seconds = seconds
        self.rate1 = rate
        self.overlap = overlap
        self.minlen = minlen

    def __len__(self):
        sum = 0
        for _, _, _, num in self.paths:
            sum += num
        return sum

    def __load_paths__(self):
        paths = []
        birds = os.listdir(self.path)
        birds = sorted(birds)

        sum = 0
        for bird in birds:
            calls = os.listdir(self.path + bird + "/")
            calls = filter(lambda x: ".ogg" or ".wav" in x, calls)
            calls = sorted(calls)
            num_calls = []
            for call in calls:
                path = self.path + bird + "/" + call
                sig, rate = openAudioFile(path=path, sample_rate=44100)
                split = splitSignal(sig, rate, 3, 0, 3)
                counter = 1
                os.remove(path)
                for elem in split:
                    new_name = self.path + bird + "/" + call[:-4] + "_" + str(counter) + ".wav"
                    sf.write(new_name, elem, rate)
                    counter += 1
                num_calls.append(sum + len(split))
                sum += len(split)
            paths += [(bird, calls, num_calls, sum)]
            print(bird)
        
        exit()
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

    def __get_spec_from_path__(self, path):
        sig, rate = openAudioFile(path, self.rate)
        x = get_spec(sig, rate)
        return x[None, :, :]

    def __getitem__(self, idx):
        path, bird = self.__idx_to_path__(idx)
        return self.__get_spec_from_path__(path), bird

print("Start loading dataset")
start = time.time()
train_dataset = CallsDataset("1dataset/birdclef/calls/test/")
stop = time.time()
print("Stop loading dataset")
print(stop-start)

print(train_dataset.__getitem__(1))