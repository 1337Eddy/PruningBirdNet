from collections import OrderedDict
from xml.etree.ElementTree import iselement
import model 
import torch 
from torch import optim 
from torch import nn 

path = "models/birdnet_pruned/birdnet_final.pt"
skip_handling = model.Skip_Handling.PADD
checkpoint = torch.load(path)
filters = checkpoint['filters']

birdnet = model.BirdNet(filters=filters, skip_handling=skip_handling)
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

torch.onnx.export(birdnet, dummy_input, "onnx/birdnet_finetuned.onnx", verbose=True, input_names=input_names, output_names=output_names)


