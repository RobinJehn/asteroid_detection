import torch
import torch.nn as nn
from typing import Tuple, List


class CNN(nn.Module):
    """
    Convolutional Neural Network that takes in an image and produces a feature vector.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int] = (30, 30),
        num_conv_blocks: int = 2,
        filters_list: List[int] = [16, 32],
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        output_size: int = 1,
    ):
        """
        Args:
            image_shape (Tuple[int, int]): Shape of the input image
            num_conv_blocks (int): Number of convolutional blocks
            filters_list (List[int]): List of number of filters for each block
            kernel_size (int): Size of the kernel
            stride (int): Stride of the convolution
            padding (int): Padding of the convolution
            output_size (int): The number of outputs
        """
        super(CNN, self).__init__()
        self.image_shape = image_shape

        self.num_conv_blocks = num_conv_blocks
        self.filters_list = filters_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_size = output_size

        self.maxpool_kernel_size = 2
        self.maxpool_stride = 2

        self.conv_blocks = nn.ModuleList(
            [nn.Conv2d(1, filters_list[0], kernel_size, stride, padding)]
        )
        self.conv_blocks.extend(
            [
                nn.Conv2d(
                    filters_list[i], filters_list[i + 1], kernel_size, stride, padding
                )
                for i in range(num_conv_blocks - 1)
            ]
        )

        # We only need one ReLU layer and one maxpool layer, as they don't learn any parameters
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(
            kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride
        )
        output_size_after_max_pools = (
            image_shape[0] // (self.maxpool_stride**self.num_conv_blocks),
            image_shape[1] // (self.maxpool_stride**self.num_conv_blocks),
        )

        self.feature_vector = nn.Linear(
            filters_list[-1]
            * output_size_after_max_pools[0]
            * output_size_after_max_pools[1],
            output_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Imaege of shape (n, 1, image_shape[0], image_shape[1])

        Returns: torch.Tensor of size (n, output_size)
        """
        for i in range(self.num_conv_blocks):
            x = self.conv_blocks[i](x)
            x = self.relu(x)
            x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.feature_vector(x)
        return x


class CFN(nn.Module):
    """
    End-to-end model that uses a CNN to classify each image and then uses an MLP to combine these outputs.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int] = (30, 30),
        num_conv_blocks: int = 2,
        conv_filters_list: List[int] = [16, 32],
        conv_kernel_size: int = 3,
        conv_stride: int = 1,
        conv_padding: int = 1,
        feature_vector_size: int = 10,
        images_per_sequence: int = 4,
        hidden_mlp_layers: int = 2,
        hidden_mlp_size: int = 64,
    ):
        """
        Args:
            image_shape (Tuple[int, int]): Shape of the input image
            num_conv_blocks (int): Number of convolutional blocks
            conv_filters_list (List[int]): List of number of filters for each block
            conv_kernel_size (int): Size of the kernel
            conv_stride (int): Stride of the convolution
            conv_padding (int): Padding of the convolution
            feature_vector_size (int): Size of the feature vector
            images_per_sequence (int): Number of images per sequence
            hidden_mlp_layers (int): Number of hidden layers in the MLP
            hidden_mlp_size (int): Size of the hidden layers in the MLP
        """
        super(CFN, self).__init__()
        self.image_shape = image_shape
        self.images_per_sequence = images_per_sequence
        self.feature_vector_size = feature_vector_size
        self.num_conv_blocks = num_conv_blocks
        self.conv_filters_list = conv_filters_list
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding

        self.hidden_mlp_layers = hidden_mlp_layers
        self.hidden_mlp_size = hidden_mlp_size

        self.cnn = CNN(
            image_shape=image_shape,
            num_conv_blocks=num_conv_blocks,
            filters_list=conv_filters_list,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            output_size=feature_vector_size,
        )

        self.mlp = MLP(
            images_per_sequence * feature_vector_size,
            1,
            hidden_size=hidden_mlp_size,
            hidden_layers=hidden_mlp_layers,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Concatenated images of shape (n, images_per_sequence, image_shape[0], image_shape[1])

        Returns: Prediction for the asteroid candidate(s) (n, 1)
        """
        # Tuple of images_per_sequence * (n, 1, image_shape[0], image_shape[1])
        images = torch.split(images, 1, dim=1)

        # Tuple of images_per_sequence * (n, feature_vector_size)
        feature_vectors = [self.cnn(image) for image in images]
        # Tensor of shape (n, images_per_sequence * feature_vector_size)
        feature_vector = torch.cat(feature_vectors, dim=1)

        x = self.mlp(feature_vector)
        return torch.sigmoid(x)


class TwoStage(nn.Module):
    """
    Two stage model that first uses a CNN to classify each image and
    then uses an MLP to combine these outputs to classify the sequence.

    This model is based on
    """

    def __init__(self, stage1: CNN, images_per_sequence: int):
        """
        Args:
            stage1 (CNN): CNN model that classifies each image
            images_per_sequence (int): Number of images per sequence
        """
        super(TwoStage, self).__init__()
        self.stage1 = stage1
        self.stage1.eval()

        # Turn of training for the CNN
        for param in self.stage1.parameters():
            param.requires_grad = False

        # MLP to combine the outputs
        mlp_input_size = self.stage1.output_size * images_per_sequence
        self.stage2 = MLP(mlp_input_size, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Concatenated images of shape (n, images_per_sequence, image_shape[0], image_shape[1])

        Returns: Prediction for the asteroid candidate(s) (n, 1)
        """
        # Tuple of images_per_sequence * (n, 1, image_shape[0], image_shape[1])
        images = torch.split(images, 1, dim=1)

        # Tuple of images_per_sequence * (n, self.stage1.output_size)
        classes = [self.stage1(image) for image in images]
        # Tensor of shape (n, images_per_sequence * self.stage1.output_size)
        output = torch.cat(classes, dim=1)

        return self.stage2(output)


# Multi Layer Perceptron
class MLP(nn.Module):
    """
    Multi Layer Perceptron with ReLU activation and sigmoid output.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        hidden_layers: int = 2,
    ):
        """
        Args:
            input_size (int): Size of the input
            output_size (int): Size of the output
            hidden_size (int): Size of the hidden layers
            hidden_layers (int): Number of hidden layers
        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        output_layer_input_size = input_size
        if hidden_layers > 0:
            self.layers.extend([nn.Linear(input_size, hidden_size)])
            self.layers.extend(
                [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
            )
            output_layer_input_size = hidden_size

        self.output_layer = nn.Linear(output_layer_input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (n, input_size)

        Returns: Output tensor of shape (n, output_size)
        """
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x
