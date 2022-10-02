import os
from subprocess import call
import subprocess
import time
import torch 
import csv



def localize_float(number):
    number = str(number)
    return number.replace('.', ',')

def determine_network_size(train_file, csv_path = "/media/eddy/7A9B-14DF/Dokumente/Studium/Vorlesungen/BA-10_SS22/Bachelor Arbeit/Trainingsplanung/size.csv"):
    file = open(train_file, "r")
    lines = file.readlines()

    for l in lines:
        command = l.strip()
        command = command.split(' ')
        start = time.time()
        call(command)
        stop = time.time()

        load_path = command[command.index('--save_path') + 1]
        dataset = command[command.index('--dataset_path') + 1]
        epochs = command[command.index('--epochs') + 1]
        seed = command[command.index('--seed') + 1]

        
        channel_multiplier = command[command.index('--channel_multiplier') + 1]
        

        minutes = (stop - start) / 60
        print(minutes)
        
        folder_elements = os.listdir(load_path)
        load_path = os.path.join(load_path, "birdnet_v" + str(len(folder_elements)-3) + ".pt")

        output = str(subprocess.check_output(["python", "analyze.py", "--mode", "test", "--load_path", load_path, "--dataset_path", dataset]))
        acc, param, flops = output[2:-3].split(' ')

        with open(csv_path, 'a') as file: 
            writer = csv.writer(file)
            writer.writerow([load_path, seed, channel_multiplier, localize_float(acc), param, flops, epochs])
            file.close()


def gamma_delta(train_file, csv_path="/media/eddy/7A9B-14DF/Dokumente/Studium/Vorlesungen/BA-10_SS22/Bachelor Arbeit/Trainingsplanung/gamma_delta.csv", dataset="1dataset/1data/calls/"):
    file = open(train_file, "r")
    lines = file.readlines()

    for l in lines:
        command = l.strip()
        command = command.split(' ')
        call(command)
        load_path = command[command.index('--save_path') + 1]
        epochs = command[command.index('--epochs') + 1]
        gamma = command[command.index('--gamma') + 1]
        delta = command[command.index('--delta') + 1]
        seed = command[command.index('--seed') + 1]
        folder_elements = os.listdir(load_path)
        load_path = os.path.join(load_path, "birdnet_v" + str(len(folder_elements)-3) + ".pt")

        output = str(subprocess.check_output(["python", "analyze.py", "--mode", "test", "--load_path", load_path, "--dataset_path", dataset]))
        acc, param, flops = output[2:-3].split(' ')

        with open(csv_path, 'a') as file: 
            writer = csv.writer(file)
            writer.writerow([load_path, seed, localize_float(delta), localize_float(gamma), localize_float(acc), param, flops, epochs])
            file.close()

def prune(train_file, csv_path="/media/eddy/7A9B-14DF/Dokumente/Studium/Vorlesungen/BA-10_SS22/Bachelor Arbeit/Trainingsplanung/BirdCLEF/prune.csv", dataset="/media/eddy/datasets/birdclef_small/"):
    file = open(train_file, "r")
    lines = file.readlines()

    for l in lines:
        command = l.strip()
        command = command.split(' ')
        call(command)
        load_path = command[command.index('--save_path') + 1]
        epochs = command[command.index('--epochs') + 1]
        channel_ratio = command[command.index('--channel_ratio') + 1]
        block_ratio = command[command.index('--block_ratio') + 1]
        temperature = command[command.index('--block_temperatur') + 1]
        seed = command[command.index('--seed') + 1]
        mode = command[command.index('--mode') + 1]

        load_path_appendix = f"pruned_c{int(float(channel_ratio) * 100)}_b{block_ratio}_{mode}_temp{temperature}_modeALL/"
        load_path += load_path_appendix
        folder_elements = os.listdir(load_path)
        load_path = os.path.join(load_path, "birdnet_v" + str(len(folder_elements)-3) + ".pt")

        output = str(subprocess.check_output(["python", "analyze.py", "--mode", "test", "--load_path", load_path, "--dataset_path", dataset]))
        acc, param, flops = output[2:-3].split(' ')
        with open(csv_path, 'a') as file: 
            writer = csv.writer(file)
            writer.writerow([load_path, seed, mode, localize_float(channel_ratio), block_ratio, localize_float(temperature), localize_float(acc), param, flops, epochs])
            file.close()
#determine_network_size("/media/eddy/7A9B-14DF/Dokumente/Studium/Vorlesungen/BA-10_SS22/Bachelor Arbeit/Trainingsplanung/train_baseline", "/media/eddy/7A9B-14DF/Dokumente/Studium/Vorlesungen/BA-10_SS22/Bachelor Arbeit/Trainingsplanung/baseline.csv")
#determine_network_size("/media/eddy/7A9B-14DF/Dokumente/Studium/Vorlesungen/BA-10_SS22/Bachelor Arbeit/Trainingsplanung/train_baseline", "/media/eddy/7A9B-14DF/Dokumente/Studium/Vorlesungen/BA-10_SS22/Bachelor Arbeit/Trainingsplanung/baseline.csv")
#gamma_delta("/media/eddy/7A9B-14DF/Dokumente/Studium/Vorlesungen/BA-10_SS22/Bachelor Arbeit/Trainingsplanung/CENT-EURO/Determine_gamma_delta")
#test_folder(command)
prune("/media/eddy/7A9B-14DF/Dokumente/Studium/Vorlesungen/BA-10_SS22/Bachelor Arbeit/Trainingsplanung/BirdCLEF/temperature_no_padd")