import torch
import numpy as np
import torch.nn as nn

class awacDrQCNNEncoder(nn.Module):
    def __init__(self, env_image_size, img_channel, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
        )

        test_mat = torch.zeros(1, img_channel, env_image_size, env_image_size)
        for conv_layer in self.cnn_layers:
            test_mat = conv_layer(test_mat)
        fc_input_size = int(np.prod(test_mat.shape))

        self.head = nn.Sequential(
            nn.Linear(fc_input_size, feature_dim),
            nn.LayerNorm(feature_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs, detach=False):
        obs = obs / 255.
        out = self.cnn_layers(obs)

        if detach:
            out = out.detach()

        out = torch.flatten(out, 1)
        out = self.head(out)
        out = torch.tanh(out)
        return out

    def output_shape(self):
        return (self.feature_dim,)

    def tie_weights(self, src, trg):
        assert type(src) == type(trg)
        trg.weight = src.weight
        trg.bias = src.bias