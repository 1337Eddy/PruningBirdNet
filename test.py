import torch 
import model 

names = [
    "/media/eddy/datasets/models/new/birdnet_final.pt",
    "/media/eddy/datasets/models/new/pruned/channel_90/block_0/pruned_c90_b0_CURL_temp0.0_modeALL/birdnet_final.pt",
    "/media/eddy/datasets/models/new/pruned/channel_90/block_0/pruned_c90_b0_CURL_temp1.0_modeALL/birdnet_final.pt",
    "/media/eddy/datasets/models/new/pruned/channel_90/block_0/pruned_c90_b0_CURL_temp2.0_modeALL/birdnet_final.pt",

    "/media/eddy/datasets/models/new/pruned/channel_70/pruned_c70_b0_CURL_temp0.0_modeALL/birdnet_final.pt",
    "/media/eddy/datasets/models/new/pruned/channel_70/pruned_c70_b0_CURL_temp1.0_modeALL/birdnet_final.pt",
    "/media/eddy/datasets/models/new/pruned/channel_70/pruned_c70_b0_CURL_temp2.0_modeALL/birdnet_final.pt",

    "/media/eddy/datasets/models/new/pruned/channel_50/pruned_c50_b0_CURL_temp0.0_modeALL/birdnet_final.pt",
    "/media/eddy/datasets/models/new/pruned/channel_50/pruned_c50_b0_CURL_temp1.0_modeALL/birdnet_final.pt",
    "/media/eddy/datasets/models/new/pruned/channel_50/pruned_c50_b0_CURL_temp2.0_modeALL/birdnet_final.pt",

    "/media/eddy/datasets/models/new/pruned/channel_30/pruned_c30_b0_CURL_temp0.0_modeALL/birdnet_final.pt",
    "/media/eddy/datasets/models/new/pruned/channel_30/pruned_c30_b0_CURL_temp1.0_modeALL/birdnet_final.pt",
    "/media/eddy/datasets/models/new/pruned/channel_30/pruned_c30_b0_CURL_temp2.0_modeALL/birdnet_final.pt",

    "/media/eddy/datasets/models/small/birdnet_final.pt",
    "/media/eddy/datasets/models/small/pruned/pruned_c90_b0_CURL_temp0.0_modeALL/birdnet_final.pt",
    "/media/eddy/datasets/models/small/pruned/pruned_c90_b0_CURL_temp1.0_modeALL/birdnet_final.pt",
    "/media/eddy/datasets/models/small/pruned/pruned_c90_b0_CURL_temp2.0_modeALL/birdnet_final.pt",
]

def print_test_accuracys(names):
    checkpoints = []

    for name in names:
        checkpoints.append((name, torch.load(name)))

    for name, checkpoint in checkpoints:
        print(name)

        print(f"1 epoch: {checkpoint['test_acc_list'][1]}")
        print(f"max: {max(checkpoint['test_acc_list'])}\n")

def print_conv_layer_scaling_factors(name):
    checkpoint = torch.load(name)
    birdnet = model.BirdNet(filters=checkpoint['filters'])
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    birdnet = birdnet.float()
    birdnet.load_state_dict(checkpoint['model_state_dict'])
    counter = 0
    sum = 0
    s = torch.nn.Softmax(dim=0)
    for resstack in birdnet.module.classifier:
        if isinstance(resstack, model.ResStack):
            for resblock in resstack.classifier:
                if isinstance(resblock, model.Resblock): 
                    fst_layer = resblock.classifier[3].weight.cpu()
                    snd_layer = resblock.classifier[7].weight.cpu()
                    #print("ResBlock")
                    print(s(resblock.W)[0])
                    #print(torch.sum(torch.abs(fst_layer)))
                    #print(torch.sum(torch.abs(snd_layer)))
                    #print()
                    sum += torch.sum(torch.abs(fst_layer))
                    sum += torch.sum(torch.abs(snd_layer))
                    counter += len(fst_layer)
                    counter += len(snd_layer)
                if isinstance(resblock, model.DownsamplingResBlock): 
                    fst_layer = resblock.classifierPath[1].weight.cpu()
                    snd_layer = resblock.classifierPath[4].weight.cpu()
                    #print()
                    last_layer = resblock.batchnorm[0].weight.cpu()
                    print("DS Block: " + str(s(resblock.W)[0]))
                    #print(torch.abs(fst_layer))
                    #print(torch.abs(snd_layer))
                    #print(last_layer)
                    #print()
                    sum += torch.sum(torch.abs(fst_layer))
                    sum += torch.sum(torch.abs(snd_layer))
                    sum += torch.sum(torch.abs(last_layer))
                    counter += len(fst_layer)
                    counter += len(snd_layer)
                    counter += len(last_layer)
                    
    if sum != 0:
        sum.cuda()
    else: 
        sum = torch.tensor([0]).cuda()  
    return sum, counter
print("\nGamma 0.3, Delta 0.3")
print_conv_layer_scaling_factors("/media/eddy/datasets/models/no_maxpool/search_gamma_delta/gamma03_delta03/birdnet_final.pt")
print("\nGamma 0.3, Delta 0.5")
print_conv_layer_scaling_factors("/media/eddy/datasets/models/no_maxpool/search_gamma_delta/gamma03_delta05/birdnet_final.pt")
print("\nGamma 0.3, Delta 0.7")
print_conv_layer_scaling_factors("/media/eddy/datasets/models/no_maxpool/search_gamma_delta/gamma03_delta07/birdnet_final.pt")
print("\nGamma 0.5, Delta 0.3")
print_conv_layer_scaling_factors("/media/eddy/datasets/models/no_maxpool/search_gamma_delta/gamma05_delta03/birdnet_final.pt")
print("\nGamma 0.5, Delta 0.5")
print_conv_layer_scaling_factors("/media/eddy/datasets/models/no_maxpool/search_gamma_delta/gamma05_delta05/birdnet_final.pt")
print("\nGamma 0.5, Delta 0.7")
print_conv_layer_scaling_factors("/media/eddy/datasets/models/no_maxpool/search_gamma_delta/gamma05_delta07/birdnet_final.pt")
print("\nGamma 0.7, Delta 0.3")
print_conv_layer_scaling_factors("/media/eddy/datasets/models/no_maxpool/search_gamma_delta/gamma07_delta03/birdnet_final.pt")
print("\nGamma 0.7, Delta 0.5")
print_conv_layer_scaling_factors("/media/eddy/datasets/models/no_maxpool/search_gamma_delta/gamma07_delta05/birdnet_final.pt")
print("\nGamma 0.7, Delta 0.7")
print_conv_layer_scaling_factors("/media/eddy/datasets/models/no_maxpool/search_gamma_delta/gamma07_delta07/birdnet_final.pt")
print()
#print_test_accuracys(names)