import os

import random
import shutil

source_path = "/media/eddy/bachelor-arbeit/PruningBirdNet/1dataset/1data/1calls/"
destination_path = "/media/eddy/bachelor-arbeit/PruningBirdNet/1dataset/1data/calls/"

folders = os.listdir(source_path)

for bird in folders:
    calls = os.listdir(source_path + bird + "/")
    calls = filter(lambda file: ".wav" in file, calls)
    calls = list(map(lambda x: x[:-4], calls))
    #print(calls)
    random.shuffle(calls)
    num_train = int(len(calls) * 0.7)
    num_test = int(len(calls) * 0.2)
    train = calls[:num_train]
    test = calls[num_train:num_train+num_test]
    val = calls[num_train+num_test:]

    os.mkdir(destination_path + "train" + "/" + bird + "/")
    os.mkdir(destination_path + "test" + "/" + bird + "/")
    os.mkdir(destination_path + "val" + "/" + bird + "/")
    for file in train:
        shutil.copyfile(source_path + bird + "/" + file + ".wav", destination_path + "train" + "/" +  bird + "/" + file + ".wav")
        shutil.copyfile(source_path + bird + "/" + file + ".meta.json", destination_path + "train" + "/" +  bird + "/" + file + ".meta.json")
    for file in test:
        shutil.copyfile(source_path + bird + "/" + file + ".wav", destination_path + "test" + "/" + bird + "/" + file + ".wav")
        shutil.copyfile(source_path + bird + "/" + file + ".meta.json", destination_path + "test" + "/" + bird + "/" + file + ".meta.json")
    for file in val:
        shutil.copyfile(source_path + bird + "/" + file + ".wav", destination_path + "val" + "/" + bird + "/" + file + ".wav")
        shutil.copyfile(source_path + bird + "/" + file + ".meta.json", destination_path + "val" + "/" + bird + "/" + file + ".meta.json")
