import torch
import torch.nn as nn

class SpectroNet(torch.nn.Module):
    # SpectroNet (architecture for spectrogram conversion piano to guitar)

    def __init__(self):
        super(SpectroNet, self).__init__()

        # Input spectrogram shape = (batch, 1, 1025, 130) = (batch, channel, freq bins (h), slices (w))

        # Params
        pool_size = (2,2)

        # Encoder conv blocks (4)
        self.e1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.e2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        )
        self.e3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.e4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        )

        # Transformer blocks (4)
        self.t1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1))
        )
        self.t2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1))
        )
        self.t3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1))
        )
        self.t4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1))
        )

        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        # Decoder deconv blocks(4)
        self.d1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=(3,3), padding=1),
            #nn.ConvTranspose2d(256, 128, kernel_size=(3,3), padding=1, stride=(2,2)),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.d2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=(3,3), padding=1),
            #nn.ConvTranspose2d(128, 64, kernel_size=(3,3), padding=1, stride=pool_size),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        )
        self.d3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=(3,3), padding=1),
            #nn.ConvTranspose2d(64, 32, kernel_size=(3,3), padding=1, stride=(2,2)),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.d4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, kernel_size=(3,3), padding=1),
            #nn.ConvTranspose2d(32, 1, kernel_size=(3,3), padding=1, stride=pool_size),
            #nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        )

        '''
        # Recurrent block
        rnn_hidden_units = params['rnn_units']
        rnn_hidden_layers = params['rnn_layers']
        feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / self.height_reduction)

        # Bidirectional RNN
        self.r1 = nn.LSTM(int(feature_dim), hidden_size=rnn_hidden_units, num_layers=rnn_hidden_layers, dropout=0.5, bidirectional=True)

        # Split embedding parameters
        self.num_notes = 1
        self.num_lengths = 1

        # Split embedding layers
        self.note_emb = nn.Linear(2 * rnn_hidden_units, self.num_notes + 1)     # +1 for blank symbol
        self.length_emb = nn.Linear(2 * rnn_hidden_units, self.num_lengths + 1) # +1 for blank symbol

        # Log Softmax at end for CTC Loss (dim = vocab dimension)
        self.sm = nn.LogSoftmax(dim=2)

        print('Vocab size:', self.num_lengths + self.num_notes)
        '''

    def forward(self, x):

        #print('before 1:', x.shape)
        x = self.e1(x)
        #print('after 1:',x.shape,'\n')

        #print('before 2:', x.shape)
        x = self.e2(x)
        #print('after 2:',x.shape,'\n')

        #print('before 3:', x.shape)
        x = self.e3(x)
        #print('after 3:',x.shape,'\n')

        #print('before 4:', x.shape)
        x = self.e4(x)
        #print('after 4:',x.shape,'\n')

        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)

        #print('after t:',x.shape)

        #print('before 1:', x.shape)
        x = self.d1(x)
        #print('after 1:',x.shape,'\n')

        #print('before 2:', x.shape)
        x = self.d2(x)
        #print('after 2:',x.shape,'\n')

        #print('before 3:', x.shape)
        x = self.d3(x)
        #print('after 3:',x.shape,'\n')

        #print('before 4:', x.shape)
        x = self.d4(x)
        #print('after 4:',x.shape,'\n')

        '''
        params = self.params
        width_reduction = self.width_reduction
        height_reduction = self.height_reduction
        input_shape = x.shape # = batch, channels, height, width
        
        # Conv blocks (4)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)

        # Prepare output of conv block for recurrent blocks
        features = x.permute(3, 0, 2, 1)  # -> [width, batch, height, channels] 
        feature_dim = params['conv_filter_n'][-1] * (params['img_height'] // height_reduction)
        feature_width = (2*2*2*input_shape[3]) // (width_reduction)
        stack = (int(feature_width), input_shape[0], int(feature_dim))
        features = torch.reshape(features, stack)  # -> [width, batch, features]

        # Recurrent block
        rnn_out, _ = self.r1(features)

        # Split embeddings
        note_out = self.note_emb(rnn_out)
        length_out = self.length_emb(rnn_out)

        # Log softmax (for CTC Loss)
        note_logits = self.sm(note_out)
        length_logits = self.sm(length_out)

        return note_logits, length_logits
        '''

        return x