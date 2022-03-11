import torch
import torch.nn as nn

class WavRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(WavRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

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
            num_layers=2, 
            batch_first=True
        )

        # Linear Transformation
        self.l1 = nn.Linear(256, 256)


        # Decoder 

        self.d1 =nn.LSTM (
            input_size=self.hidden_size, 
            hidden_size=self.input_size, 
            num_layers=1, 
            batch_first=True
        )

        self.d2 = nn.LSTM(
            input_size=self.input_size, 
            hidden_size =self.input_size, 
            num_layers=2, 
            batch_first=True
        )


    def forward(self, x):

        #print(x.shape)
        x = torch.reshape(x, (x.shape[0], int(x.shape[2]/self.input_size), self.input_size))

        # LSTM encoder
        x, hc1 = self.e1(x)
        x, hc1 = self.e2(x)
      
        # Linear transformer
        x = self.l1(x)

        # LSTM decoder
        x, hc2 = self.d1(x)
        x, hc2 = self.d2(x)

        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]*x.shape[2]))

        return x
        