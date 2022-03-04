import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np

#This source file is based on the NGCF framwork published by Xiang Wang et al.
#We would like to thank and offer our appreciation to them.
#Original algorithm can be found in paper: Neural Graph Collaborative Filtering, SIGIR 2019.

def _init_weights(m):
    # classname = m.__class__.__name__
    # if classname.find('Linear') != -1:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.zero_(m.bias)

class NGCFDL(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, u_name_emb, i_name_emb):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = 768
        self.u_name_emb = u_name_emb
        self.i_name_emb = i_name_emb
        self.weight_size = weight_size
        self.n_layers = len(self.weight_size)
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()

        self.weight_size = [self.embedding_dim] + self.weight_size
        for i in range(self.n_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.dropout_list.append(nn.Dropout(dropout_list[i]))

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # self.user_name = nn.Embedding(n_users, 768) # fc to (n_users, 16) then aggregate with user_embedding
        # self.item_name  = nn.Embedding(n_items, 768)

        # self.user_name = nn.Sequential(
        #     nn.Linear(768, 2)
        # )

        # self.item_name = nn.Sequential(
        #     nn.Linear(768, 2)
        # )


        self._init_weight_()

    def _init_weight_(self):
        # nn.init.xavier_uniform_(self.user_embedding.weight)
        # nn.init.xavier_uniform_(self.item_embedding.weight)
        # self.user_name.apply(_init_weights)
        # self.item_name.apply(_init_weights)
        self.user_embedding.weight.data = self.u_name_emb[:self.n_users]
        self.item_embedding.weight.data = self.i_name_emb[:self.n_items]

    def forward(self, adj):
        # user_name_emb = self.user_name(user_name_input)
        # item_name_emb = self.item_name(item_name_input)
        # print("user_name_emb")
        # print(user_name_emb.size())
        # print(self.user_embedding.embedding_dim)

        # agg_user_emb = torch.cat((self.user_embedding, user_name_emb), 1)
        # agg_item_emb = torch.cat((self.item_embedding, item_name_emb), 1)
        # print("agg_user_emb")
        # print(agg_user_emb.size())
        # print("agg_user_emb")
        # print(agg_item_emb.size())
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        # print("ego_embeddings")
        # print(ego_embeddings.size())
        # exit()
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            # here remove the bi_embeddings can remove the inner product of e_i and e_u;
            ego_embeddings = sum_embeddings + bi_embeddings
            #ego_embeddings = sum_embeddings 
            ego_embeddings = self.dropout_list[i](ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings

