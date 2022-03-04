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
        self.embedding_dim = 32
        self.grec_dim = 16
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

        self.user_embedding = nn.Embedding(n_users, self.grec_dim)
        self.item_embedding = nn.Embedding(n_items, self.grec_dim)
        
        
        # self.user_name = nn.Embedding(n_users, 768) # fc to (n_users, 16) then aggregate with user_embedding
        # self.item_name  = nn.Embedding(n_items, 768)

        # self.user_name = nn.Sequential(
        #     nn.Linear(768, 2)
        # )

        # self.item_name = nn.Sequential(
        #     nn.Linear(768, 2)
        # )
        

        self._init_weight_()
        
        
        # user_emb_proj = nn.Linear(self.user_embedding, n_users, bias=False)
        # user_emb_proj.weight = self.user_embedding.weight

        self.item_emb_encoder = nn.Sequential(
            nn.Linear(768, 32, bias=False),
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.5),

            nn.Linear(32, 16, bias=False)
        )

        self.user_emb_encoder = nn.Sequential(
            nn.Linear(768, 32, bias=False),
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.5),

            nn.Linear(32, 16, bias=False)
        )

        # self.item_emb_proj.weight = self.item_embedding.weight
        # self.encoder = nn.Sequential(
        #     nn.Linear(768, 16), # fc1
        #     self.item_emb_proj
        # )
    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        # self.user_name.apply(_init_weights)
        # self.item_name.apply(_init_weights)
        # self.user_embedding.weight.data = self.u_name_emb[:self.n_users]
        # self.item_embedding.weight.data = self.i_name_emb[:self.n_items]

    def set_item_embedding(self):
        # self.item_embedding.weight.data[16:] = self.item_embedding_created
        # print("self.item_embedding.weight.data")
        # print(self.item_embedding.weight.data.size())
        # print("self.item_embedding_created")
        # print(self.item_embedding_created.size())
        return torch.cat((self.item_embedding.weight.data, self.item_embedding_created), dim=1)

    def set_user_embedding(self):
        # self.item_embedding.weight.data = self.item_embedding_created
        # print("self.user_embedding.weight.data")
        # print(self.user_embedding.weight.data.size())
        # print("self.user_embedding_created")
        # print(self.user_embedding_created.size())
        return torch.cat((self.user_embedding.weight.data, self.user_embedding_created), dim=1)
        

    def forward(self, adj, user_name_embs, sent_embs):
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
        # print(sent_embs)
        # print(self.item_emb_encoder)
        self.user_embedding_created = self.user_emb_encoder(user_name_embs)
        self.item_embedding_created = self.item_emb_encoder(sent_embs)
        
        self.agg_item_embedding = self.set_item_embedding()
        self.agg_user_embedding = self.set_user_embedding()

        # print("agg_item_embedding")
        # print(self.agg_item_embedding.size())
        # ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding_created), dim=0)
        ego_embeddings = torch.cat((self.agg_user_embedding, self.agg_item_embedding), dim=0)

        # print("ego_embeddings")
        # print(ego_embeddings.size())
        # exit()
        
        all_embeddings = [ego_embeddings]
        # print("all_embeddings[0]")
        # print(all_embeddings[0].size())
        for i in range(self.n_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            # print("side_embeddings")
            # print(side_embeddings.size())
            # print(adj.detach().cpu())
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
            # print("sum_embeddings")
            # print(sum_embeddings.size())
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            # print("bi_embeddings")
            # print(bi_embeddings.size())
            # here remove the bi_embeddings can remove the inner product of e_i and e_u;
            ego_embeddings = sum_embeddings + bi_embeddings
            #ego_embeddings = sum_embeddings 
            ego_embeddings = self.dropout_list[i](ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings

