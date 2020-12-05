import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
"""
reference: https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
code: https://github.com/mdcramer/deep-learning/blob/master/seq2seq/sequence_to_sequence_implementation.ipynb
"""

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers, dropout=0):
        super().__init__()
        # input_dim = 10000
        self.emb_dim = emb_dim #15
        self.hid_dim = hid_dim #50
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        # nn.LSTMCell: single LSTM, nn.LSTM: stack of multiple LSTM
        self.multirnn = nn.LSTM(emb_dim, hid_dim, num_layers, dropout=dropout)

    def forward(self, x):
        """
        :param x: [batch_size, max_src_length]
        :return hidden: [[batch_size, max_src_length, hid_dim]
                cell: [[batch_size, max_src_length, hid_dim]
        """

        embeded = self.embedding(x) # [batch_size, max_src_length, emb_size]
        outputs, (hidden, cell) = self.multirnn(embeded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, num_layers, dropout=0):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.multirnn = nn.LSTM(emb_dim, hid_dim, num_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.multirnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[0, :] # <GO> tokens
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else top1
        return outputs

# if __name__ == "__main__":


