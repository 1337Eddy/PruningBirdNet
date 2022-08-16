import argparse
import torch

def flop_conv_layer(channels_in, channels_out, kernel, width, height):
    return ((channels_in * kernel[0] * kernel[1]) + (channels_in * kernel[0] * (kernel[1] - 1) - 1)) * channels_out * width * height, channels_out


def calc_flops(filters, kernels, input_width, input_height, channels=1):
    width = input_width
    height = input_height
    sum = 0
    #Input layer
    result, channels = flop_conv_layer(channels, filters[0][0][0], (5,5), width, height)
    sum += result
    for i in range(1, len(filters) - 1):
        #Downsampling Block
        result, channels = flop_conv_layer(channels, filters[i][0][0], (1,1), width, height)
        sum += result
        sum += 2* channels

        result, channels = flop_conv_layer(channels, filters[i][0][1], (3,3), width, height)
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

            result, channels = flop_conv_layer(channels, filters[i][j][0], (3,3), width, height)
            sum += result
            sum += 2* channels

            result, channels = flop_conv_layer(channels, filters[i][j][1], (3,3), width, height)
            sum += result
            sum += 2* channels
            sum += 2*skip

    #Classification Path
    result, channels = flop_conv_layer(channels, filters[-1][0], (4,12), width, height)
    width = 1
    height = 13
    sum += result
    result, channels = flop_conv_layer(channels, filters[-1][1], (1,1), width, height)
    sum += result
    result, channels = flop_conv_layer(channels, filters[-1][2], (1,1), width, height)
    sum += result
    

    return sum 




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/media/eddy/datasets/models/new_shape/delta03_gamma04/birdnet_v20.pt', help='path to file')
    parser.add_argument('--width', default=384)
    parser.add_argument('--height', default=64)
    kernel_sizes=[(5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
    args = parser.parse_args()
    checkpoint = torch.load(args.path)
    width = args.width
    height = args.height
    print(calc_flops(checkpoint['filters'], kernel_sizes, width, height))
if __name__ == '__main__':
    main()
