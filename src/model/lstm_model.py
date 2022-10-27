import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """
    A multi-layer perceptron with "ReLU" activation function
    """

    def __init__(self, input_size, output_size, hidden_size):
        """
        Args:
            input_size (int): number of inputs
            output_size (int): number of outputs
            hidden_size (list): a list containing the number of neurons in all hidden layers
        """

        super().__init__()
        self.n_hidden_layers = len(hidden_size)
        self.fc = nn.ModuleList()

        for i in range(self.n_hidden_layers+1):
            # Input layer
            if i == 0:
                self.fc.append(nn.Linear(input_size, hidden_size[i]))
            else:
                # Output layer
                if i == self.n_hidden_layers:
                    self.fc.append(nn.Linear(hidden_size[i-1], output_size))
                else:
                    # Hidden layer
                    self.fc.append(nn.Linear(hidden_size[i-1], hidden_size[i]))

    def forward(self, x):
        x1 = F.relu(self.fc[0](x))
        for i in range(1, self.n_hidden_layers+1):
            x1 = F.relu(self.fc[i](x1))

        return x1


class RluModel(nn.Module):
    """
    MLP -> LSTM -> MLP
    """

    def __init__(self, input_size, output_size_1, hidden_size_1, hidden_size_2, hidden_size_lstm, num_layers_lstm):
        """

        Args:
            input_size (int): number of inputs
            output_size_1 (int): number of outputs of the first MLP
            hidden_size_1 (list): a list containing the number of neurons in all hidden layers of the first MLP
            hidden_size_2 (list): a list containing the number of neurons in all hidden layers of the second MLP
            hidden_size_lstm (int): number of features in hidden and cell states of the LTSM
            num_layers_lstm (int): number LTSM layers
        """
        super().__init__()

        self.mlp_1 = Mlp(input_size, output_size_1, hidden_size_1)
        self.lstm = nn.LSTM(output_size_1, hidden_size_lstm, num_layers_lstm)
        self.mlp_2 = Mlp(hidden_size_lstm, 1, hidden_size_2)

    def forward(self, sq_in):
        """

        Args:
            sq_in (torch.FloatTensor): input sequence

        Returns:
            torch.FloatTensor: RUL of all steps in the input sequence

        """

        # seq_in has a shape of (T x M) in which T is the sequence length and M is the number of sensor values
        # seq_in                -> MLP1   ->  out_mlp_1
        # (T x M)               -> MLP1   ->  (T x output_size_1)
        out_mlp_1 = self.mlp_1(sq_in)

        # output_mlp_1          -> LSTM   -> out_lstm
        # (T x output_size_1)   -> LSTM   -> (T x hidden_size_lstm)
        out_lstm, (_h, _c) = self.lstm(out_mlp_1)

        # out_lstm               -> MLP2 -> rul
        # (T x hidden_size_lstm) -> MLP2 -> (T x 1)
        rul = self.mlp_2(out_lstm)

        return rul

