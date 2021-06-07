"""
@author "Laszlo Szoke
This file is used to contain the different models.
"""
from torch import nn


class SimpleMLP(nn.Module):

    def __init__(self, input_size,
                 output_size,
                 hidden_size=128):
        """
        This class represents a Simple MLP network for Policy Gradient applications.
        :param input_size: size of the CNN input channels.
        :param output_size: Size of the output.
        :param hidden_size:
        :param num_layers:
        """
        super(SimpleMLP, self).__init__()
        self.name = "SimpleMLP"

        # Building network modules
        self.linear = nn.Sequential(nn.Linear(in_features=input_size,
                                              out_features=hidden_size),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(in_features=hidden_size,
                                              out_features=hidden_size // 2),
                                    nn.BatchNorm1d(hidden_size // 2),
                                    nn.ReLU(),
                                    nn.Linear(in_features=hidden_size // 2, out_features=output_size)
                                    )

    def forward(self, x):
        """
        Function for forwarding the input of the network
        Parameters
        ----------
        x: has the dimensions as [time, x, y, channel]

        Returns
        -------
        the output of the network

        """
        # Getting the output of the MLP as the probability of actions
        linear_result = self.linear(x)
        return linear_result


class Convolutional_LSTM_MLP(nn.Module):

    def __init__(self, in_channels,
                 output_size,
                 hidden_channels=16,
                 lstm_input_size=20,
                 lstm_hidden_size=64,
                 num_layers=1,
                 kernel_size=11,
                 padding=5,
                 stride=1):
        """
        This class represents a CNN --> LSTM --> MLP network for Policy Gradient applications.
        :param in_channels: size of the CNN input channels.
        :param output_size: Size of the output.
        :param hidden_channels:
        :param lstm_input_size:
        :param lstm_hidden_size:
        :param num_layers:
        :param kernel_size:
        :param padding:
        :param stride:
        """
        # todo: save parameters and load them back if needed.
        super(Convolutional_LSTM_MLP, self).__init__()
        self.name = "CNN_LSTM_MPL"
        # Building network modules
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels * 2,
                      kernel_size=kernel_size, padding=padding,
                      stride=stride),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels * 2, out_channels=hidden_channels * 2,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels * 2, out_channels=hidden_channels,
                      kernel_size=11, padding=5, stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(in_channels=hidden_channels, out_channels=1,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(1),
            nn.AdaptiveMaxPool2d(output_size=(lstm_input_size // 2, 2))
        )
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=num_layers)

        self.linear = nn.Sequential(nn.Linear(in_features=lstm_hidden_size,
                                              out_features=lstm_hidden_size // 2),
                                    nn.ReLU(),
                                    nn.Linear(in_features=lstm_hidden_size // 2, out_features=output_size)
                                    )

    def forward(self, x):
        """
        Function for forwarding the input of the network
        Parameters
        ----------
        x: has the dimensions as [time, x, y, channel]

        Returns
        -------
        the output of the network

        """
        # Going through the timesteps and extracting the convolutional output
        lstm_input = self.convolution(x).flatten(-2, -1)  # using that batch size is one.
        # Preparing and propagating through lstm layer(s)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(lstm_input)
        # Removing unnecessary time steps (only using the last one)
        x = x[-1, :, :]
        # Getting the output of the MLP as the probability of actions
        linear_result = self.linear(x)
        return linear_result
