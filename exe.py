from re import sub
from subprocess import call 
import os
import torch

def train_from_file(name):
    file = open(name, "r")
    lines = file.readlines()

    for l in lines:
        command = l.strip()
        call(command.split(' '))

def test_folder(path, dataset="1dataset/1data/calls/"):
    dir = os.listdir(path)
    dir = list(sorted(dir))
    for elem in dir:
        sub_dir = os.listdir(os.path.join(path, elem))
        sub_dir.remove("birdnet_final.pt")
        sub_dir.remove("birdnet_raw.pt")
        sub_dir = sorted(sub_dir)
        file = sub_dir[-1]
        print(elem)
        call(["python", "analyze.py", "--mode", "test", "--load_path", os.path.join(path, elem, file), "--dataset_path", dataset])
        call(["python", "flop_calc.py", "--path", os.path.join(path, elem, file)])
        print()
def test_file(file, dataset="1dataset/1data/calls/"):
    print(file)
    call(["python", "analyze.py", "--mode", "test", "--load_path", file, "--dataset_path", dataset])
    call(["python", "flop_calc.py", "--path", file])
    print()

def print_train_phase(file):
    checkpoint = torch.load(file)
    acc = checkpoint['test_acc_list']
    string = ""
    for i in range(0, len(acc)):
        string += "(" + str(i) + ", " + str(round(acc[i], 3)) + ") "
    print(string)

#test_file("/media/eddy/datasets/models/SSS_hard_temp/baseline_BirdCLEF/pruned/Temp/0.0/pruned_c70_b0_CURL_temp0.0_modeALL/birdnet_v3.pt", dataset="/media/eddy/datasets/birdclef_small/")

#train_from_file("/home/eddy/Trainingsplanung.txt")
#
print_train_phase("/media/eddy/datasets/models/SSS_hard_temp/baseline_BirdCLEF/small/birdnet_final.pt")
print()
print_train_phase("/media/eddy/datasets/models/SSS_hard_temp/baseline_CentEuro-NFC/small/birdnet_final.pt")
