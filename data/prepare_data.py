import time

import os
import soundfile as sf
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/media/eddy/bachelor-arbeit/PruningBirdNet/utils')

from audio import openAudioFile, splitSignal

def split_data_in_slices(directory):
    paths = []
    birds = os.listdir(directory)
    birds = sorted(birds)
    sum = 0
    for bird in birds:
        calls = os.listdir(directory + bird + "/")
        calls = filter(lambda x: ".ogg" or ".wav" in x, calls)
        calls = sorted(calls)
        num_calls = []
        for call in calls:
            path = directory + bird + "/" + call
            sig, rate = openAudioFile(path=path, sample_rate=44100)
            split = splitSignal(sig, rate, 3, 0, 3)
            counter = 1
            os.remove(path)
            for elem in split:
                new_name = directory + bird + "/" + call[:-4] + "_" + str(counter) + ".wav"
                sf.write(new_name, elem, rate)
                counter += 1
            num_calls.append(sum + len(split))
            sum += len(split)
        paths += [(bird, calls, num_calls, sum)]
        print(bird)

print("Start loading dataset")
start = time.time()
split_data_in_slices("/media/eddy/datasets/birdclef/")
stop = time.time()
print("Stop loading dataset")
print(stop-start)
