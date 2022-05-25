from xml.etree.ElementTree import iselement
import model 
import torch 
from torch import optim 
from torch import nn 

path = "models/paddtest/birdnet_final.pt"
skip_handling = model.Skip_Handling.PADD

birdnet = model.BirdNet(skip_handling=skip_handling)
birdnet = birdnet.float()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(birdnet.parameters(), lr=0.001) 
checkpoint = torch.load(path)


dummy_input = torch.randn(16, 1, 64, 512)


input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(birdnet, dummy_input, "birdnet.onnx", verbose=True, input_names=input_names, output_names=output_names)


