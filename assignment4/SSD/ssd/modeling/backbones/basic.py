import torch
from typing import Tuple, List


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self, output_channels: List[int],image_channels: int, output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        self.map1 = torch.nn.Sequential(
            torch.torch.nn.Conv2d(image_channels, 32, 3, 1, 1),
            torch.nn.MaxPool2d(2,2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.MaxPool2d(2,2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64,3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, output_channels[0], 3, 2, 1)
        )

        self.map2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels[0], 128, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, output_channels[1], 3, 2, 1)
        )

        self.map3 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels[1], 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, output_channels[2], 3, 2, 1)
        )

        self.map4 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels[2], 128, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, output_channels[3], 3, 2, 1)
        )

        self.map5 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels[3], 128, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, output_channels[4], 3, 2, 1)
        )

        self.map6 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels[4], 128, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, output_channels[5], 3, 1, 0)
        )


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        x = self.map1(x)
        out_features.append(x)
        x = self.map2(x)
        out_features.append(x)
        x = self.map3(x)
        out_features.append(x)
        x = self.map4(x)
        out_features.append(x)
        x = self.map5(x)
        out_features.append(x)
        x = self.map6(x)
        out_features.append(x)

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

