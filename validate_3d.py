
import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
import math
import torch.nn.functional as F
import numpy as np

class ConvTranspose3d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:[int, tuple],
                 stride:[int, tuple] = 1,
                 padding:[int, tuple] = 0,
                 output_padding:[int, tuple] = 0,
                 groups:int = 1,
                 bias=False,
                 #padding_mode:str ='zeros',
                 dilation:[int, tuple] = 1):
      

        super(ConvTranspose3d, self).__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        dilation = _triple(dilation)
        padding = _triple(padding)
        output_padding = _triple(output_padding) 
        #print(dilation)
        #print(padding)
        #print(kernel_size)

        assert all(op < st or op < dl for op, st, dl in zip(output_padding, stride, dilation)), 'output padding must be smaller than either stride or dilation, got output padding={}, dilation={}, stride={}'.format(output_padding, dilation, stride)

        input_padding = tuple(d*(ks-1)-p for d, p, ks in zip(dilation, padding, kernel_size))

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        # Assertions for constructor arguments
        assert len(kernel_size) == 3, '3D kernel size expected!'
        assert len(stride) == 3, '3D Stride size expected!!'
        assert len(padding) == 3, '3D Padding size expected!!'
        assert len(dilation) == 3, '3D dilation size expected!'
        assert groups == 1, 'Groups other than 1 not yet implemented!'

        # Store constructor arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_padding = input_padding
        self.output_padding = output_padding
        self.dilation = dilation

        self.groups = groups

        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

        ################## Validation ##################
        self.official_conv3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        self.official_conv3d.weight = self.weight
        self.official_conv3d.bias = self.bias


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        # Define shortcut names for dimensions of input and kernel
        (Batch, _, l_i, h_i, w_i) = tuple(input.shape)
        (l_k, h_k, w_k) = self.kernel_size
        (l_p, h_p, w_p) = self.padding
        (l_ip, h_ip, w_ip) = self.input_padding
        (l_op, h_op, w_op) = self.output_padding
        (l_d, h_d, w_d) = self.dilation
        (l_s, h_s, w_s) = self.stride

        # Compute the size of the output tensor based on the zero padding
        l_o = (l_i - 1) * l_s - 2 * l_p + l_d * (l_k - 1) + l_op + 1
        h_o = (h_i - 1) * h_s - 2 * h_p + h_d * (h_k - 1) + h_op + 1
        w_o = (w_i - 1) * w_s - 2 * w_p + w_d * (w_k - 1) + w_op + 1
        #print("{} {} {}".format(l_o, h_o, w_o))

        # Pre-define output tensors
        out = torch.zeros(Batch, self.out_channels, l_o, h_o, w_o).to(input.device)

        zero_feed = torch.zeros(*input[:, :, 0, :, :].shape).to(input.device)

        # Convolve each kernel frame i with each input frame j
        for i in range(0, l_k):
            # Calculate the zero-offset of kernel frame i
            zero_offset = - (l_p) + i
            # Calculate the range of input frame j corresponding to kernel frame i
            # Convolve each kernel frame i with corresponding input frame j
            for j in range(0, l_i):
                # Calculate the output frame
                out_frame = l_s * j + zero_offset
                if out_frame < 0 or out_frame >= out.shape[2]:
                  #print("{} -> {} (no)".format((i,l_s * j), out_frame))
                  continue
                #print("{} -> {}".format((i,l_s * j), out_frame))
                # Add results to this output frame
                out[:, :, out_frame, :, :] += F.conv_transpose2d(input=input[:, :, j, :, :],
                                                       weight=self.weight[:, :, i, :, :],
                                                       bias=None,
                                                       stride=self.stride[1::],
                                                       padding=self.padding[1::],
                                                       output_padding=self.output_padding[1::],
                                                       dilation=self.dilation[1::],
                                                       groups=self.groups)

        # Add bias to output
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)
        
        ################### Validation ##################
        out_official = self.official_conv3d(input)

        delta = torch.max(abs(out_official - out))
        return delta
        
