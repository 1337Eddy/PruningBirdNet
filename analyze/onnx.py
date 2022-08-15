from collections import OrderedDict
import os
import torch 
from torch import optim 
from torch import nn 

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/media/eddy/bachelor-arbeit/PruningBirdNet/')
import model 

def export_onnx(path, save_path):
    skip_handling = model.Dim_Handling.PADD
    checkpoint = torch.load(path)
    filters = checkpoint['filters']

    birdnet = model.BirdNet(filters=filters, dimension_handling=skip_handling)
    #birdnet = torch.nn.DataParallel(birdnet).cuda()
    birdnet = birdnet.float()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(birdnet.parameters(), lr=0.001) 

    model_state_dict = checkpoint['model_state_dict']
    new_model_state_dict = OrderedDict()

    for elem in model_state_dict.keys():
        new_model_state_dict[elem[7:]] = model_state_dict[elem]

    birdnet.load_state_dict(new_model_state_dict)


    dummy_input = torch.randn(16, 1, 64, 512)


    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]
    #print(birdnet)
    pytorch_total_params = sum(p.numel() for p in birdnet.parameters())
    pytorch_trainable_params = sum(p.numel() for p in birdnet.parameters() if p.requires_grad)

    #summary(birdnet, (1, 64, 512))
    torch.onnx.export(birdnet, dummy_input, save_path, verbose=True, input_names=input_names, output_names=output_names)

def generate_onnx_for_all(path):
    elements = os.listdir(path)
    folders = filter(lambda elem: os.path.isdir(path + elem + "/"), elements)
    for folder in folders:
        generate_onnx_for_all(path + folder + "/")
    if "birdnet_final.pt" in elements:
        #plot_hist(path + "birdnet_final.pt", path + "hist")
        #plot_train_phase(path + "birdnet_final.pt", path + "train_phase")
        export_onnx(path + "birdnet_final.pt", path + "model.onnx")


generate_onnx_for_all("/media/eddy/datasets/models/new/pruned/channel_90/block_0/")
