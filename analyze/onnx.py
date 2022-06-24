from collections import OrderedDict
from xml.etree.ElementTree import iselement
import model 
import torch 
from torch import optim 
from torch import nn 
from torchsummary import summary

path = "models/birdnet/pruned/test3/birdnet_final.pt"
skip_handling = model.Skip_Handling.PADD
checkpoint = torch.load(path)
filters = checkpoint['filters']

birdnet = model.BirdNet(filters=filters, skip_handling=skip_handling)
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
torch.onnx.export(birdnet, dummy_input, "onnx/channel_pruned_no_padd.onnx", verbose=True, input_names=input_names, output_names=output_names)


