import torch
from torch import nn


__all__ = ['SimpleVGGCNN']


class SimpleVGGCNN(nn.Module):
    """
        REF: Model architecture soured from:(Tiny VGG) Fang, Xing. (2017).
        Understanding deep learning via backtracking and deconvolution.
        Journal of Big Data. 4. 40. 10.1186/s40537-017-0101-8.
        
        Difference: PReLU layers, automatic calculation of convolution layers 
                    output.
    """

    def __init__(self, image_height, image_width, channels, hidden_units, output_units):
        super(SimpleVGGCNN, self).__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=hidden_units,
                      kernel_size=4,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.PReLU(num_parameters=hidden_units),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=4,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.PReLU(num_parameters=hidden_units),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.hidden_block = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=4,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.PReLU(num_parameters=hidden_units),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=4,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.PReLU(num_parameters=hidden_units),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.conv_out_size = self. _get_conv_out_size(image_height, image_width, channels)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.conv_out_size,
                      out_features=output_units)
        )

    def _get_conv_out_size(self, image_height, image_width, channels):
        dummy_input = torch.zeros(1, channels, image_height, image_width,)
        output = self.input_block(dummy_input)
        output = self.hidden_block(output)

        return int(torch.prod(torch.tensor(output.size()[1:])))

    def forward(self, input):
        _output1 = self.input_block(input)
        _output2 = self.hidden_block(_output1)
        output = self.classifier(_output2)
        return output
