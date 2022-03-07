import torch
import torch.nn as nn

class WavRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(WavRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Encoder
        self.e1 = nn.LSTM(
            input_size=self.input_size, 
            hidden_size =self.input_size, 
            num_layers=1, 
            batch_first=True
        )
        self.e2 = nn.LSTM(
            input_size=self.input_size, 
            hidden_size =self.hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        # Transformation
        self.t1 = nn.Sequential(
            nn.Conv1d(258, 258, kernel_size=3, padding=1),
            nn.BatchNorm1d(258),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t2 = nn.Sequential(
            nn.Conv1d(258, 258, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(258),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t3 = nn.Sequential(
            nn.Conv1d(258, 258, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(258),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t4 = nn.Sequential(
            nn.Conv1d(258, 258, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(258),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder 
        self.d1 = nn.LSTM(
            input_size=self.hidden_size, 
            hidden_size =self.input_size, 
            num_layers=1, 
            batch_first=True
        )

        self.d2 =nn.LSTM (
            input_size=self.input_size, 
            hidden_size=self.input_size, 
            num_layers=1, 
            batch_first=True
        )


    def forward(self, x):

        #print(x.shape)
        x = torch.reshape(x, (x.shape[0], int(x.shape[2]/self.input_size), self.input_size))

        # LSTM encoder
        x, hc1 = self.e1(x)
        x, hc1 = self.e2(x)

        # Convolutional transformer
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        #x = self.t4(x)

        # LSTM decoder
        x, hc2 = self.d1(x)
        x, hc2 = self.d2(x)

        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]*x.shape[2]))

        return x
        