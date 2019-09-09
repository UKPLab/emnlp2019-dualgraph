#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
from torch_geometric.data import Data, Batch

def get_batch_gnn_td(batch, embeddings):

    use_cuda = batch.src[0].is_cuda

    node1_index = batch.node1[0].data.transpose(0, 1)  # [ seqLen x batch_size ] ==> [ batch_size x seqLen ]
    node2_index = batch.node2[0].data.transpose(0, 1)

    node1_voc = batch.dataset.fields['node1'].base_field.vocab.itos
    node2_voc = batch.dataset.fields['node2'].base_field.vocab.itos
    label_voc = batch.dataset.fields['src'].base_field.vocab.itos

    sent_mask_original = torch.lt(torch.eq(batch.src[0][:, :, 0].data, 1), 1)  # [ seqLen x batch_size ]
    sent_mask = sent_mask_original.permute(1, 0).contiguous()
    sent_mask = autograd.Variable(torch.FloatTensor(sent_mask.tolist()), requires_grad=False)
    sent_mask_original = autograd.Variable(torch.FloatTensor(sent_mask_original.tolist()), requires_grad=False)

    if use_cuda:
        sent_mask = sent_mask.cuda()
        sent_mask_original = sent_mask_original.cuda()

    embeddings = embeddings * sent_mask.unsqueeze(2)

    edges_index = []
    for d, de in enumerate(node1_index):
        edge_index_1 = []
        edge_index_2 = []
        for a, arc in enumerate(de):
            arc_0 = label_voc[node1_index[d, a]]
            arc_1 = label_voc[node2_index[d, a]]

            if arc_0 == '<unk>' or arc_0 == '<pad>' or arc_0 == '<blank>':
                pass
            elif arc_1 == '<unk>' or arc_1 == '<pad>' or arc_1 == '<blank>':
                pass
            else:
                arc_1 = int(node1_voc[arc])
                arc_2 = int(node2_voc[node2_index[d, a]])

                edge_index_1.append(arc_2)
                edge_index_2.append(arc_1)

        edges_index.append(torch.tensor([edge_index_1, edge_index_2], dtype=torch.long))

    assert len(edges_index) == embeddings.data.shape[0]
    list_geometric_data = []
    for idx, emb in enumerate(embeddings):

        data = Data(x=emb, edge_index=edges_index[idx])
        list_geometric_data.append(data)

    return Batch.from_data_list(list_geometric_data).to('cuda'), sent_mask_original

def get_batch_gnn_bu(batch, embeddings):

    use_cuda = batch.src[0].is_cuda

    node1_index = batch.node1[0].data.transpose(0, 1)
    node2_index = batch.node2[0].data.transpose(0, 1)

    node1_voc = batch.dataset.fields['node1'].base_field.vocab.itos
    node2_voc = batch.dataset.fields['node2'].base_field.vocab.itos
    label_voc = batch.dataset.fields['src'].base_field.vocab.itos

    sent_mask_original = torch.lt(torch.eq(batch.src[0][:, :, 0].data, 1), 1)
    sent_mask = sent_mask_original.permute(1, 0).contiguous()
    sent_mask = autograd.Variable(torch.FloatTensor(sent_mask.tolist()), requires_grad=False)
    sent_mask_original = autograd.Variable(torch.FloatTensor(sent_mask_original.tolist()), requires_grad=False)

    if use_cuda:
        sent_mask = sent_mask.cuda()
        sent_mask_original = sent_mask_original.cuda()

    embeddings = embeddings * sent_mask.unsqueeze(2)

    edges_index = []
    for d, de in enumerate(node1_index):
        edge_index_1 = []
        edge_index_2 = []
        for a, arc in enumerate(de):
            arc_0 = label_voc[node1_index[d, a]]
            arc_1 = label_voc[node2_index[d, a]]

            if arc_0 == '<unk>' or arc_0 == '<pad>' or arc_0 == '<blank>':
                pass
            elif arc_1 == '<unk>' or arc_1 == '<pad>' or arc_1 == '<blank>':
                pass
            else:
                arc_1 = int(node1_voc[arc])
                arc_2 = int(node2_voc[node2_index[d, a]])

                edge_index_1.append(arc_1)
                edge_index_2.append(arc_2)

        edges_index.append(torch.tensor([edge_index_1, edge_index_2], dtype=torch.long))

    assert len(edges_index) == embeddings.data.shape[0]
    list_geometric_data = []
    for idx, emb in enumerate(embeddings):

        data = Data(x=emb, edge_index=edges_index[idx])
        list_geometric_data.append(data)

    return Batch.from_data_list(list_geometric_data).to('cuda'), sent_mask_original
