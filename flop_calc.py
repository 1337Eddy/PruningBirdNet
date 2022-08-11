import torch

def flop_conv_layer(channels_in, channels_out, kernel, width, height):
    return ((channels_in * kernel[0] * kernel[1]) + (channels_in * kernel[0] * (kernel[1] - 1) - 1)) * channels_out * width * height, channels_out


filters = [[[32]], 
        [[32, 32, 64], [64, 64], [64, 64], [64, 64]], 
        [[64, 64, 128], [128, 128], [128, 128], [128, 128]], 
        [[128, 128, 256], [256, 256], [256, 256], [256, 256]], 
        [[256, 256, 512], [512, 512], [512, 512], [512, 512]],
        [512, 512, 83]]
kernel_sizes=[(5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]

def calc_flops(filters, kernels, input_width, input_height, channels=1):
    width = input_width
    height = input_height
    sum = 0
    #Input layer
    result, channels = flop_conv_layer(channels, filters[0][0][0], kernels[0], width, height)
    sum += result
    for i in range(1, len(filters) - 1):
        #Downsampling Block
        result, channels = flop_conv_layer(channels, filters[i][0][0], (1,1), width, height)
        sum += result
        sum += 2* channels

        result, channels = flop_conv_layer(channels, filters[i][0][1], kernels[i-1], width, height)
        sum += result
        sum += 2* channels

        width /= 2
        height /= 2
        result, channels = flop_conv_layer(channels, filters[i][0][2], (1,1), width, height)
        sum += 2 * result
        skip = channels * width * height
        sum += 2 * skip
        sum += 2* channels


        for j in range(1, len(filters[i])):
            #Residual Block
            skip = channels * width * height
            sum += 2* channels

            result, channels = flop_conv_layer(channels, filters[i][j][0], kernels[i-1], width, height)
            sum += result
            sum += 2* channels

            result, channels = flop_conv_layer(channels, filters[i][j][1], kernels[i-1], width, height)
            sum += result
            sum += 2* channels
            sum += 2*skip

    #Classification Path
    result, channels = flop_conv_layer(channels, filters[-1][0], (4,10), width, height)
    width = 1
    height = 23
    sum += result
    result, channels = flop_conv_layer(channels, filters[-1][1], (1,1), width, height)
    sum += result
    result, channels = flop_conv_layer(channels, filters[-1][2], (1,1), width, height)
    sum += result
    

    return sum 
checkpoint_30 = torch.load("/media/eddy/datasets/models/new/pruned/channel_30/pruned_c30_b0_CURL_temp0.0_modeALL/birdnet_final.pt")    
checkpoint_50 = torch.load("/media/eddy/datasets/models/new/pruned/channel_50/pruned_c50_b0_CURL_temp0.0_modeALL/birdnet_final.pt")    
checkpoint_70 = torch.load("/media/eddy/datasets/models/new/pruned/channel_70/pruned_c70_b0_CURL_temp0.0_modeALL/birdnet_final.pt")    
checkpoint_90 = torch.load("/media/eddy/datasets/models/new/pruned/channel_90/pruned_c90_b0_CURL_temp0.0_modeALL/birdnet_raw.pt")    

width = 512
height = 64

print(calc_flops(filters, kernel_sizes, width, height))
print(calc_flops(checkpoint_30['filters'], kernel_sizes, width, height))
print(calc_flops(checkpoint_50['filters'], kernel_sizes, width, height))
print(calc_flops(checkpoint_70['filters'], kernel_sizes, width, height))
print(calc_flops(checkpoint_90['filters'], kernel_sizes, width, height))
