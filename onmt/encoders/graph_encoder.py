import torch
import torch.nn as nn
from onmt.encoders.encoder import EncoderBase
from onmt.mymodules.utils_gnn import get_batch_gnn_td, get_batch_gnn_bu
from torch_geometric.nn import GATConv, GINConv, GatedGraphConv
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn import Sequential, Linear, ReLU


class GraphEncoder(EncoderBase):

    def __init__(self, embeddings, word_vec_size, gnn_type, gnn_layers, rnn_size):
        super(GraphEncoder, self).__init__()

        self.is_graph_encoder = True

        self.gnn_type = gnn_type
        self.gnn_layers = gnn_layers
        self.embeddings = embeddings
        self.dropout = nn.Dropout(0.3)
        self.num_inputs = word_vec_size
        self.rnn_size = rnn_size

        if self.gnn_type == 'ggnn':
            self.gnn_td = GatedGraphConv(self.num_inputs, self.gnn_layers)
            self.gnn_bu = GatedGraphConv(self.num_inputs, self.gnn_layers)
        elif self.gnn_type == 'gat':
            self.gnn_td = GATConv(self.num_inputs, self.num_inputs, heads=self.gnn_layers, concat=False, dropout=0.3)
            self.gnn_bu = GATConv(self.num_inputs, self.num_inputs, heads=self.gnn_layers, concat=False, dropout=0.3)
        else:
            self.gins_td = []
            self.gins_bu = []
            num_layers = self.gnn_layers
            nn_td = Sequential(Linear(self.num_inputs, self.num_inputs), ReLU(),
                             Linear(self.num_inputs, self.num_inputs))
            nn_tb = Sequential(Linear(self.num_inputs, self.num_inputs), ReLU(),
                             Linear(self.num_inputs, self.num_inputs))
            for x in range(num_layers):
                gin = GINConv(nn_td)
                self.gins_td.append(gin.cuda())
                gin = GINConv(nn_tb)
                self.gins_bu.append(gin.cuda())

        self.bilstm = nn.LSTM(self.rnn_size, self.rnn_size // 2, num_layers=2, bidirectional=True, dropout=0.3)

        if self.gnn_type == 'gin':
            self.layers_seq = nn.Sequential(*self.gins_td, *self.gins_bu, self.bilstm)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            embeddings,
            opt.word_vec_size,
            opt.gnn_type,
            opt.gnn_layers,
            opt.rnn_size)

    def forward(self, src, lengths, batch):

        # [seqLen x batch_size]
        embeddings = self.embeddings(src)
        embeddings = embeddings.permute(1, 0, 2).contiguous()  # [t, b, h] -> [b, t, h]

        batch_size = embeddings.size()[0]
        seq_len = embeddings.size()[1]
        dim_emb = embeddings.size()[2]

        if self.gnn_type != 'gin':

            batch_geometric, sent_mask = get_batch_gnn_td(batch, embeddings)
            memory_bank = batch_geometric.x
            memory_bank = self.gnn_td(memory_bank, batch_geometric.edge_index)  # [t, b, h]
            memory_bank = memory_bank.view((batch_size, seq_len, dim_emb))  # [t * b, h] -> # [b, t, h]
            memory_bank = memory_bank.permute(1, 0, 2).contiguous() # [b, t, h] -> # [t, b, h]
            memory_bank = memory_bank * sent_mask.unsqueeze(2)
            memory_bank_1 = memory_bank.permute(1, 0, 2).contiguous()  # [t, b, h] -> # [b, t, h]

            batch_geometric, sent_mask = get_batch_gnn_bu(batch, embeddings)
            memory_bank = batch_geometric.x
            memory_bank = self.gnn_bu(memory_bank, batch_geometric.edge_index)  # [t, b, h]
            memory_bank = memory_bank.view((batch_size, seq_len, dim_emb))  # [t * b, h] -> # [b, t, h]
            memory_bank = memory_bank.permute(1, 0, 2).contiguous() # [b, t, h] -> # [t, b, h]
            memory_bank = memory_bank * sent_mask.unsqueeze(2)
            memory_bank_2 = memory_bank.permute(1, 0, 2).contiguous()  # [t, b, h] -> # [b, t, h]


        else:

            memory_bank_1 = embeddings
            memory_bank_2 = embeddings

            for gnn_td, gnn_bu in zip(self.gins_td, self.gins_bu):
                batch_geometric, sent_mask = get_batch_gnn_td(batch, memory_bank_1)
                memory_bank = batch_geometric.x
                memory_bank = self.dropout(gnn_td(memory_bank, batch_geometric.edge_index))  # [t, b, h]
                memory_bank = memory_bank.view((batch_size, seq_len, dim_emb))  # [t * b, h] -> # [b, t, h]
                memory_bank = memory_bank.permute(1, 0, 2).contiguous()  # [b, t, h] -> # [t, b, h]
                memory_bank = memory_bank * sent_mask.unsqueeze(2)
                memory_bank_1 = memory_bank.permute(1, 0, 2).contiguous()  # [t, b, h] -> # [b, t, h]

                batch_geometric, sent_mask = get_batch_gnn_bu(batch, memory_bank_2)
                memory_bank = batch_geometric.x
                memory_bank = self.dropout(gnn_bu(memory_bank, batch_geometric.edge_index))  # [t, b, h]
                memory_bank = memory_bank.view((batch_size, seq_len, dim_emb))  # [t * b, h] -> # [b, t, h]
                memory_bank = memory_bank.permute(1, 0, 2).contiguous()  # [b, t, h] -> # [t, b, h]
                memory_bank = memory_bank * sent_mask.unsqueeze(2)
                memory_bank_2 = memory_bank.permute(1, 0, 2).contiguous()  # [t, b, h] -> # [b, t, h]

        memory_bank = torch.cat([memory_bank_1, memory_bank_2, embeddings], dim=2)  # [b, t, h]
        memory_bank = memory_bank.permute(1, 0, 2).contiguous()  # [b, t, h] -> # [t, b, h]
        memory_bank = memory_bank * sent_mask.unsqueeze(2)

        packed_emb = memory_bank
        if lengths is not None:
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(memory_bank, lengths_list)

        memory_bank = packed_emb
        memory_bank, (h__1, h__2) = self.bilstm(memory_bank)

        if lengths is not None:
            memory_bank = unpack(memory_bank)[0]

        return (h__1, h__2), memory_bank, lengths
