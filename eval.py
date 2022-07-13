import numpy as np
from utils import audio
import os

import torch
threshold = 0.5



class EvalBirdnet():
    def __init__(self, birdnet, dataset):
        self.dataset = dataset

        torch.cuda.manual_seed(1337)
        torch.manual_seed(73)
        self.birdnet = birdnet


    def evalFile(self, path, rate=44100, seconds=3, overlap=0, minlen=3, batchsize=16):
        sig, rate = audio.openAudioFile(path, rate)
        specs = audio.specsFromSignal(sig, rate, seconds=seconds, overlap=overlap, minlen=minlen)
        counter = 0
        time_0 = 0
        predictions = []
        while (True):
            try:
                if (counter == 0):
                    batch = next(specs)
                    batch = batch[None, None, :, :]
                    counter +=1
                elif (counter < batchsize):
                    spec = next(specs)
                    batch = torch.cat((batch, spec[None, None, :, :]), dim=0)     
                    counter +=1 
                else: 
                    batch = batch.cuda(non_blocking=True)
                    output = self.birdnet(batch.float())   
                    output = torch.squeeze(output)
                    for pred in output:
                        estimation = self.softmax(np.array(pred.cpu().detach()))
                        index = np.argmax(estimation)
                        if estimation[index] > threshold:
                            prediction = self.dataset.id_to_label(index)
                            predictions += [(time_0, time_0 + seconds, prediction, estimation[index])]
                            time_0 += seconds - overlap
                        else: 
                            predictions += [(time_0, time_0 + seconds, 'no call', None)]
                            time_0 += seconds - overlap
                    counter = 0
                    batch = None
                
            except StopIteration:
                try:
                    if (batch != None):
                        batch = batch.cuda(non_blocking=True)
                        output = self.birdnet(batch.float())   
                        output = torch.squeeze(output)
                        if (np.shape(output) == torch.Size([self.dataset.num_classes])):
                            output = output[None, :]
                        for pred in output:
                            estimation = self.softmax(np.array(pred.cpu().detach()))
                            index = np.argmax(estimation)
                            if estimation[index] > threshold:
                                prediction = self.dataset.id_to_label(index)
                                predictions += [(time_0, time_0 + seconds, prediction, estimation[index])]
                                time_0 += seconds - overlap
                            else: 
                                predictions += [(time_0, time_0 + seconds, 'no call', None)]
                                time_0 += seconds - overlap
                        break
                except:
                    print(f"File {path} ist shorter than {seconds} seconds and can't be evaluated")
                    break
        return predictions


    def eval(self, path, rate=44100, seconds=3, overlap=0, minlen=3, batchsize=16):
        self.birdnet.eval()
        if os.path.isfile(path):
            return self.evalFile(path=path, rate=rate, seconds=seconds, overlap=overlap, minlen=minlen, batchsize=batchsize)
        elif os.path.isdir(path):
            files = os.listdir(path)
            files = filter(lambda x: ".wav" in x, files)
            predictions = []
            for file in files:
                result = self.evalFile(path=path + file, rate=rate, seconds=seconds, overlap=overlap, minlen=minlen, batchsize=batchsize)
                predictions.append((file, result))
            return predictions
        else:
            print("Error")
            exit()
    
    def softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x
