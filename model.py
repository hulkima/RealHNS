import numpy as np
import torch
import ipdb
import torch.nn.functional as F
from torch import Tensor
import math
import os
import io
import copy
import time
import random
    
class EarlyStopping_onetower:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, version='SASRec_V3', verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_performance = None
        self.early_stop = False
        self.ndcg_max = None
        self.save_epoch = None
        self.delta = delta
        self.version = version

    def __call__(self, epoch, model, result_path, t_test):

        if self.ndcg_max is None:
            self.ndcg_max = t_test[2]
            self.best_performance = t_test
            self.save_epoch = epoch
            self.save_checkpoint(epoch, model, result_path, t_test)
        elif t_test[2] < self.ndcg_max:
            self.counter += 1
            print(f'In the epoch: {epoch}, EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_performance = t_test
            self.save_epoch = epoch
            self.save_checkpoint(epoch, model, result_path, t_test)
            self.counter = 0

    def save_checkpoint(self, epoch, model, result_path, t_test):
#         if self.version == 'SASRec_V3':
        if self.version == 'SASRec_V13':
            print(f'Validation loss in {epoch} decreased {self.ndcg_max:.4f} --> {t_test[2]:.4f}.  Saving model ...\n')
            with io.open(result_path + 'save_model.txt', 'a', encoding='utf-8') as file:
                file.write("NDCG@10 in epoch {} decreased {:.4f} --> {:.4f}, the HR@10 is {:.4f}, the AUC is {:.4f}, the loss_rec is {:.4f}, distance_mix_source: {:.4f}, distance_mix_target: {:.4f}, distance_source_target: {:.4f}. Saving model...\n".format(epoch, self.ndcg_max, t_test[2], t_test[7], t_test[10], t_test[11], t_test[12], t_test[13], t_test[14]))
        else:
            print(f'Validation loss in {epoch} decreased {self.ndcg_max:.4f} --> {t_test[2]:.4f}.  Saving model ...\n')
            with io.open(result_path + 'save_model.txt', 'a', encoding='utf-8') as file:
                file.write("NDCG@10 in epoch {} decreased {:.4f} --> {:.4f}, the HR@10 is {:.4f}, the AUC is {:.4f}, the loss_rec is {:.4f}. Saving model...\n".format(epoch, self.ndcg_max, t_test[2], t_test[7], t_test[10], t_test[11]))
#         elif self.version == 'SASRec_V5':
#             print(f'Validation loss in {epoch} decreased {self.ndcg_max:.4f} --> {t_test[0]:.4f}.  Saving model ...\n')
#             with io.open(result_path + 'save_model.txt', 'a', encoding='utf-8') as file:
#                 file.write("NDCG in epoch {} decreased {:.4f} --> {:.4f}, the HR is {:.4f}, the AUC is {:.4f}, the loss_rec is {:.4f} and the loss_cl1 is {:.4f}. Saving model...\n".format(epoch, self.ndcg_max, t_test[0], t_test[1], t_test[2], t_test[3], t_test[4]))
        torch.save(model.state_dict(), os.path.join(result_path, 'checkpoint.pt')) 
        self.ndcg_max = t_test[2]
        
      

    
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py
    
    
class SASRec_Embedding(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec_Embedding, self).__init__()

        self.item_num = item_num # 3416
        self.dev = args.device #'cuda'

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0) #Embedding(3417, 50, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE Embedding(200, 50)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate) #Dropout(p=0.2)

        self.attention_layernorms = torch.nn.ModuleList() # 2 layers of LayerNorm
        self.attention_layers = torch.nn.ModuleList() # 2 layers of MultiheadAttention
        self.forward_layernorms = torch.nn.ModuleList() # 2 layers of LayerNorm
        self.forward_layers = torch.nn.ModuleList() # 2 layers of PointWiseFeedForward

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) # LayerNorm(torch.Size([50]), eps=1e-08, elementwise_affine=True)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) #LayerNorm(torch.Size([50]), eps=1e-08, elementwise_affine=True)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate, batch_first=True) # MultiheadAttention((out_proj): NonDynamicallyQuantizableLinear(in_features=50, out_features=50, bias=True))
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) # LayerNorm((50,), eps=1e-08, elementwise_affine=True)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5 # torch.Size([128, 200, 64])
        positions = torch.tile(torch.arange(0,log_seqs.shape[1]), [log_seqs.shape[0],1]).cuda() # torch.Size([128, 200])
            # add the position embedding
        seqs += self.pos_emb(positions) 
        seqs = self.emb_dropout(seqs) # torch.Size([128, 200, 64])

            # mask the noninteracted position
        timeline_mask = torch.BoolTensor(log_seqs.cpu() == 0).cuda() # (128,200)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality, 200
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device='cuda')) #(200,200)

        for i in range(len(self.attention_layers)):
#             seqs = torch.transpose(seqs, 0, 1) # torch.Size([200, 128, 50])
            Q = self.attention_layernorms[i](seqs) #torch.Size([128, 200, 50])
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask) # torch.Size([128, 200, 50])
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs # torch.Size([128, 200, 50])
#             seqs = torch.transpose(seqs, 0, 1) # torch.Size([128, 200, 50])

            seqs = self.forward_layernorms[i](seqs) # torch.Size([128, 200, 50])
            seqs = self.forward_layers[i](seqs) # torch.Size([128, 200, 50])
            seqs *=  ~timeline_mask.unsqueeze(-1) # torch.Size([128, 200, 50])

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        return log_feats

    def forward(self, log_seqs): # for training      
        log_feats = self.log2feats(log_seqs) # torch.Size([128, 200, 50]) user_ids hasn't been used yet


        return log_feats # pos_pred, neg_pred    
    

class SASRec_V2_Adaptive(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec_V2_Adaptive, self).__init__()
    
        self.sasrec_embedding_source = SASRec_Embedding(item_num, args)
        self.sasrec_embedding_target = SASRec_Embedding(item_num, args)
        self.dev = args.device #'cuda'

        # for the both domain
        self.log_feat_map1 = torch.nn.Linear(args.hidden_units * 2, args.hidden_units)
        self.log_feat_map2 = torch.nn.Linear(args.hidden_units, args.hidden_units)

        self.leakyrelu = torch.nn.LeakyReLU()
        
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=2)
        self.temperature = args.temperature
        self.fname = args.dataset
        self.source_weight = args.source_weight
        self.similar_for_big = args.similar_for_big
        self.item_num = item_num
        self.interval = args.interval

    
    def forward(self, user_ids, source_log_seqs, target_log_seqs, pos_seqs, neg_seqs_list, user_train_source_sequence_for_target_indices): # for training     
#         ipdb.set_trace()
        neg_embs = []
        if self.fname == 'amazon_game':
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats = self.sasrec_embedding_target(target_log_seqs) # torch.Size([128, 200, 64])
            
#             ipdb.set_trace()
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]

            concatenate_log_feats_time = torch.cat([source_log_feats_time, target_log_feats], dim=-1)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats_time)))

            pos_embs = self.sasrec_embedding_target.item_emb(pos_seqs) # torch.Size([128, 200, 64])
            for i in range(0,len(neg_seqs_list)):
                neg_embs.append(self.sasrec_embedding_target.item_emb(neg_seqs_list[i]))

        elif self.fname == 'amazon_toy':
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats = self.sasrec_embedding_source(target_log_seqs) # torch.Size([128, 200, 64])
#             ipdb.set_trace()
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
    
    

            concatenate_log_feats = torch.cat([source_log_feats_time, target_log_feats], dim=-1)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats))) # torch.Size([128, 200, 64])

            pos_embs = self.sasrec_embedding_source.item_emb(pos_seqs) # torch.Size([128, 200, 64])
            for i in range(0,len(neg_seqs_list)):
                neg_embs.append(self.sasrec_embedding_source.item_emb(neg_seqs_list[i]))
            
        # get the l2 norm for the both domains recommendation
        log_feats_l2norm = torch.nn.functional.normalize(log_feats, p=2, dim=-1)
        pos_embs_l2norm = torch.nn.functional.normalize(pos_embs, p=2, dim=-1)
        pos_logits = (log_feats_l2norm * pos_embs_l2norm).sum(dim=-1) # torch.Size([128, 200])
        pos_logits = pos_logits * self.temperature
        neg_logits = []
        for i in range(0,len(neg_seqs_list)):
            neg_embs_l2norm_i = torch.nn.functional.normalize(neg_embs[i], p=2, dim=-1)
            neg_logits_i = (log_feats_l2norm * neg_embs_l2norm_i).sum(dim=-1) # torch.Size([128, 200])
            neg_logits_i = neg_logits_i * self.temperature
            neg_logits.append(neg_logits_i)
            
        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, source_log_seqs, target_log_seqs, item_indices): # for inference
            # user_ids: (1,)
            # log_seqs: (1, 200)
            # item_indices: (101,)e
#         ipdb.set_trace()
        if self.fname == 'amazon_game':
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([1, 200, 50]) 
            target_log_feats = self.sasrec_embedding_target(target_log_seqs) # torch.Size([1, 200, 50]) 
            
            concatenate_log_feats = torch.cat([source_log_feats[:,-1,:], target_log_feats[:,-1,:]], dim=-1)
            final_feat = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))

            item_embs = self.sasrec_embedding_target.item_emb(item_indices) 

        elif self.fname == 'amazon_toy':
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats = self.sasrec_embedding_source(target_log_seqs) # torch.Size([128, 200, 64])

            concatenate_log_feats = torch.cat([source_log_feats[:,-1,:], target_log_feats[:,-1,:]], dim=-1)
            final_feat = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))

            item_embs = self.sasrec_embedding_source.item_emb(item_indices) 
            

        # get the l2 norm for the both domains recommendation
#         final_feat = log_feats[:, -1, :] # torch.Size([1, 50]) 
        
        final_feat_l2norm = torch.nn.functional.normalize(final_feat, p=2, dim=-1)
        item_embs_l2norm = torch.nn.functional.normalize(item_embs, p=2, dim=-1)

        logits = item_embs_l2norm.matmul(final_feat_l2norm.unsqueeze(-1)).squeeze(-1) 
        logits = logits * self.temperature

        return logits # preds # (U, I)
    
    def predict_withembedding(self, user_ids, source_log_seqs, target_log_seqs, item_indices): # for inference
        if self.fname == 'amazon_game':
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([1, 200, 50]) 
            target_log_feats = self.sasrec_embedding_target(target_log_seqs) # torch.Size([1, 200, 50]) 
            
            concatenate_log_feats = torch.cat([source_log_feats[:,-1,:], target_log_feats[:,-1,:]], dim=-1)
            final_feat = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))

            item_embs = self.sasrec_embedding_target.item_emb(item_indices) 

        elif self.fname == 'amazon_toy':
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats = self.sasrec_embedding_source(target_log_seqs) # torch.Size([128, 200, 64])

            concatenate_log_feats = torch.cat([source_log_feats[:,-1,:], target_log_feats[:,-1,:]], dim=-1)
            final_feat = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))

            item_embs = self.sasrec_embedding_source.item_emb(item_indices) 
            

        # get the l2 norm for the both domains recommendation
#         final_feat = log_feats[:, -1, :] # torch.Size([1, 50]) 
        
        final_feat_l2norm = torch.nn.functional.normalize(final_feat, p=2, dim=-1)
        item_embs_l2norm = torch.nn.functional.normalize(item_embs, p=2, dim=-1)

        logits = item_embs_l2norm.matmul(final_feat_l2norm.unsqueeze(-1)).squeeze(-1) 
        logits = logits * self.temperature

        return logits, final_feat_l2norm, item_embs_l2norm # torch.Size([1, 100])
    
    def predict_final_user(self, user_ids, source_log_seqs, target_log_seqs, item_indices): # for inference
            # user_ids: (1,)
            # log_seqs: (1, 200)
            # item_indices: (101,)e
#         ipdb.set_trace()
        if self.fname == 'amazon_game':
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([1, 200, 50]) 
            target_log_feats = self.sasrec_embedding_target(target_log_seqs) # torch.Size([1, 200, 50]) 
            
            concatenate_log_feats = torch.cat([source_log_feats[:,-1,:], target_log_feats[:,-1,:]], dim=-1)
            final_feat = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))

            item_embs = self.sasrec_embedding_target.item_emb(item_indices) 

        elif self.fname == 'amazon_toy':
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats = self.sasrec_embedding_source(target_log_seqs) # torch.Size([128, 200, 64])

            concatenate_log_feats = torch.cat([source_log_feats[:,-1,:], target_log_feats[:,-1,:]], dim=-1)
            final_feat = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))

            item_embs = self.sasrec_embedding_source.item_emb(item_indices) 
            

        # get the l2 norm for the both domains recommendation
#         final_feat = log_feats[:, -1, :] # torch.Size([1, 50]) 
        
        final_feat_l2norm = torch.nn.functional.normalize(final_feat, p=2, dim=-1)
        item_embs_l2norm = torch.nn.functional.normalize(item_embs, p=2, dim=-1)

        logits = item_embs_l2norm.matmul(final_feat_l2norm.unsqueeze(-1)).squeeze(-1) 
        logits = logits * self.temperature

        return logits, final_feat_l2norm # preds # (U, I)
            
    def calculate_embedding(self, source_log_seqs, target_log_seqs, user_train_source_sequence_for_target_indices):
        if self.fname == 'amazon_toy':
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats = self.sasrec_embedding_source(target_log_seqs) # torch.Size([128, 200, 64])
            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))
            
        elif self.fname == 'amazon_game':
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats = self.sasrec_embedding_target(target_log_seqs) # torch.Size([128, 200, 64])
            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats_time = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats_time)))
            
        log_feats_l2norm = torch.nn.functional.normalize(log_feats, p=2, dim=-1)

        return log_feats_l2norm

    def calculate_score(self, source_log_seqs, target_log_seqs, user_train_source_sequence_for_target_indices,item_list):
        log_feats_l2norm = self.calculate_embedding(source_log_seqs, target_log_seqs, user_train_source_sequence_for_target_indices)

#         ipdb.set_trace() 
        if self.fname == 'amazon_toy':
            item_all_embedding = self.sasrec_embedding_source.item_emb(item_list)# torch.Size([37868, 64])
        elif self.fname == 'amazon_game':
            item_all_embedding = self.sasrec_embedding_target.item_emb(item_list) # torch.Size([11735, 64])
            
        log_feats_l2norm = log_feats_l2norm[0,-1,:].unsqueeze(0).expand([item_list.shape[0],-1]) 

        item_all_embedding_l2norm = torch.nn.functional.normalize(item_all_embedding, p=2, dim=-1)
        scores = (log_feats_l2norm*item_all_embedding_l2norm).sum(dim=-1) * self.temperature # torch.Size([11735])

        return scores
    
    
    def calculate_embedding_withembedding(self, source_log_seqs, target_log_seqs, user_train_source_sequence_for_target_indices):
        if self.fname == 'amazon_toy':
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats = self.sasrec_embedding_source(target_log_seqs) # torch.Size([128, 200, 64])
            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))
            
        elif self.fname == 'amazon_game':
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats = self.sasrec_embedding_target(target_log_seqs) # torch.Size([128, 200, 64])
            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats_time = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats_time)))
            
        log_feats_l2norm = torch.nn.functional.normalize(log_feats, p=2, dim=-1)
        source_log_feats_l2norm = torch.nn.functional.normalize(source_log_feats_time[:,-1,:], p=2, dim=-1) # torch.Size([1, 64])
        target_log_feats_l2norm = torch.nn.functional.normalize(target_log_feats[:,-1,:], p=2, dim=-1) # torch.Size([1, 64])

        return log_feats_l2norm, source_log_feats_l2norm, target_log_feats_l2norm

    def calculate_score_withembedding(self, source_log_seqs, target_log_seqs, user_train_source_sequence_for_target_indices,item_list):
        log_feats_l2norm, source_log_feats_l2norm, target_log_feats_l2norm = self.calculate_embedding_withembedding(source_log_seqs, target_log_seqs, user_train_source_sequence_for_target_indices)
#         ipdb.set_trace() 
        if self.fname == 'amazon_toy':
            item_all_embedding = self.sasrec_embedding_source.item_emb(item_list)# torch.Size([37868, 64])
        elif self.fname == 'amazon_game':
            item_all_embedding = self.sasrec_embedding_target.item_emb(item_list) # torch.Size([11735, 64])
            
        log_feats_l2norm = log_feats_l2norm[0,-1,:].unsqueeze(0).expand([item_list.shape[0],-1]) 

        item_all_embedding_l2norm = torch.nn.functional.normalize(item_all_embedding, p=2, dim=-1)
        scores = (log_feats_l2norm*item_all_embedding_l2norm).sum(dim=-1) * self.temperature # torch.Size([11735])

        return scores, source_log_feats_l2norm, target_log_feats_l2norm
    
    
    
    def calculate_random_item_user_rep(self, source_log_seqs, user_train_source_sequence_for_target_indices, pos_target):
        if self.fname == 'amazon_toy':
#             source_log_feats = torch.empty([128,200,64], dtype=torch.float32, device='cuda').uniform_(0,1)
            source_log_seqs = torch.randint(low=self.interval, high=self.item_num, size=[128,200], device='cuda', requires_grad=False) # torch.Size([1015, 10])
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])

            target_log_feats = torch.zeros(source_log_feats.shape, dtype=torch.float32, device='cuda') # torch.Size([128, 200, 64])

            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))
                        
        elif self.fname == 'amazon_game':
#             source_log_feats = torch.empty([128,200,64], dtype=torch.float32, device='cuda').uniform_(0,1)
            source_log_seqs = torch.randint(low=0, high=self.interval, size=[128,200], device='cuda', requires_grad=False) # torch.Size([1015, 10])
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([128, 200, 64])

            target_log_feats = torch.zeros(source_log_feats.shape, dtype=torch.float32, device='cuda') # torch.Size([128, 200, 64])            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats_time = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats_time)))
            
        log_feats = log_feats[torch.where(pos_target!=0)]
        log_feats_l2norm = torch.nn.functional.normalize(log_feats, p=2, dim=-1)
        return log_feats_l2norm
        
    def calculate_random_user_rep(self, source_log_seqs, user_train_source_sequence_for_target_indices, pos_target):
        if self.fname == 'amazon_toy':
            source_log_feats = torch.empty([128,200,64], dtype=torch.float32, device='cuda').uniform_(0,1)
            target_log_feats = torch.zeros(source_log_feats.shape, dtype=torch.float32, device='cuda') # torch.Size([128, 200, 64])

            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))
                        
        elif self.fname == 'amazon_game':
            source_log_feats = torch.empty([128,200,64], dtype=torch.float32, device='cuda').uniform_(0,1)
            target_log_feats = torch.zeros(source_log_feats.shape, dtype=torch.float32, device='cuda') # torch.Size([128, 200, 64])            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats_time = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats_time)))
            
        log_feats = log_feats[torch.where(pos_target!=0)]
        log_feats_l2norm = torch.nn.functional.normalize(log_feats, p=2, dim=-1)
        return log_feats_l2norm
    
    def calculate_score_source(self, user_id, source_log_seqs, user_train_source_sequence_for_target_indices,item_list, user_id_for_cluster_id, cluster_id_for_finaldim_embedding):      
#         ipdb.set_trace()
        if self.fname == 'amazon_toy':
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            
            cluster_candidate = user_id_for_cluster_id[user_id] # torch.Size([128, 1])
            target_log_feats = torch.index_select(cluster_id_for_finaldim_embedding, dim=0, index=cluster_candidate.squeeze()) # torch.Size([128, 64])
#             target_log_feats = target_embedding_candidate.unsqueeze(1).expand(source_log_feats.shape)
  
        elif self.fname == 'amazon_game':
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([128, 200, 64])
            cluster_candidate = user_id_for_cluster_id[user_id] # torch.Size([128, 1])
            target_log_feats = torch.index_select(cluster_id_for_finaldim_embedding, dim=0, index=cluster_candidate.squeeze()) # torch.Size([128, 64])
#             target_log_feats = target_embedding_candidate.unsqueeze(1).expand(source_log_feats.shape)
            
        source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
        concatenate_log_feats = torch.cat([source_log_feats_time[:,-1,:], target_log_feats], dim=-1)
        log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))
        
        log_feats_l2norm = torch.nn.functional.normalize(log_feats, p=2, dim=-1)
        
#         ipdb.set_trace() 
        if self.fname == 'amazon_toy':
            item_all_embedding = self.sasrec_embedding_source.item_emb(item_list)# torch.Size([37868, 64])
        elif self.fname == 'amazon_game':
            item_all_embedding = self.sasrec_embedding_target.item_emb(item_list) # torch.Size([11735, 64])
            
        log_feats_l2norm = log_feats_l2norm.expand([item_list.shape[0],-1]) 

        item_all_embedding_l2norm = torch.nn.functional.normalize(item_all_embedding, p=2, dim=-1)
        scores = (log_feats_l2norm*item_all_embedding_l2norm).sum(dim=-1) * self.temperature # torch.Size([11735])

        return scores
    
    
    def calculate_score_source_simple(self, source_log_seqs, user_train_source_sequence_for_target_indices, item_list):      
#         ipdb.set_trace()
        if self.fname == 'amazon_toy':
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats = torch.zeros(source_log_feats.shape, dtype=torch.float32, device='cuda') # torch.Size([128, 200, 64])
            item_all_embedding = self.sasrec_embedding_source.item_emb(item_list)# torch.Size([37868, 64])
   
        elif self.fname == 'amazon_game':
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats = torch.zeros(source_log_feats.shape, dtype=torch.float32, device='cuda') # torch.Size([128, 200, 64])
            item_all_embedding = self.sasrec_embedding_target.item_emb(item_list) # torch.Size([11735, 64])

        source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
        concatenate_log_feats = torch.cat([source_log_feats_time[:,-1,:], target_log_feats[:,-1,:]], dim=-1)
        log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))
        
        log_feats_l2norm = torch.nn.functional.normalize(log_feats, p=2, dim=-1)
        log_feats_l2norm = log_feats_l2norm.expand([item_list.shape[0],-1]) 

        item_all_embedding_l2norm = torch.nn.functional.normalize(item_all_embedding, p=2, dim=-1)
        scores = (log_feats_l2norm*item_all_embedding_l2norm).sum(dim=-1) * self.temperature # torch.Size([11735])

        return scores
    
    

    

    def calculate_itemembedding(self, item_candidate):
        if self.fname == 'amazon_toy':
            target_neg_embedding = self.sasrec_embedding_source.item_emb(item_candidate) # torch.Size([128, 200, 1000, 64])
        elif self.fname == 'amazon_game':
            target_neg_embedding = self.sasrec_embedding_target.item_emb(item_candidate) # torch.Size([128, 100, 200, 64])

        target_neg_embedding = torch.nn.functional.normalize(target_neg_embedding, p=2, dim=-1)
        return target_neg_embedding
    
    
    def calculate_source_embedding_simple(self, source_log_seqs, user_train_source_sequence_for_target_indices, pos_target):
        if self.fname == 'amazon_toy':
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats = torch.zeros(source_log_feats.shape, dtype=torch.float32, device='cuda') # torch.Size([128, 200, 64])
            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))
                        
        elif self.fname == 'amazon_game':
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats = torch.zeros(source_log_feats.shape, dtype=torch.float32, device='cuda') # torch.Size([128, 200, 64])            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats_time = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats_time)))
            
        log_feats = log_feats[torch.where(pos_target!=0)]
        log_feats_l2norm = torch.nn.functional.normalize(log_feats, p=2, dim=-1)
        return log_feats_l2norm
        
        
        
        
    
    def calculate_source_embedding_cluster(self, user_id, source_log_seqs, target_log_seqs, user_train_source_sequence_for_target_indices, pos_target, user_id_for_cluster_id, cluster_id_for_finaldim_embedding):
#         ipdb.set_trace()
        if self.fname == 'amazon_toy':
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            
            cluster_candidate = torch.index_select(user_id_for_cluster_id, dim=0, index=user_id) # torch.Size([128, 1])
            target_embedding_candidate = torch.index_select(cluster_id_for_finaldim_embedding, dim=0, index=cluster_candidate.squeeze()) # torch.Size([128, 64])
            target_log_feats = target_embedding_candidate.unsqueeze(1).expand(source_log_feats.shape)
            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats_time = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats_time)))
                        
        elif self.fname == 'amazon_game':
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([128, 200, 64])
            cluster_candidate = torch.index_select(user_id_for_cluster_id, dim=0, index=user_id) # torch.Size([128, 1])
            target_embedding_candidate = torch.index_select(cluster_id_for_finaldim_embedding, dim=0, index=cluster_candidate.squeeze()) # torch.Size([128, 64])
            target_log_feats = target_embedding_candidate.unsqueeze(1).expand(source_log_feats.shape)
            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats_time = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats_time)))
            
        log_feats = log_feats[torch.where(pos_target!=0)]
        log_feats_l2norm = torch.nn.functional.normalize(log_feats, p=2, dim=-1)
        return log_feats_l2norm
    
    def calculate_source_embedding_cluster_all(self, user_id, source_log_seqs, target_log_seqs, user_train_source_sequence_for_target_indices, pos_target, user_id_for_cluster_id, cluster_id_for_finaldim_embedding):
#         ipdb.set_trace()
        if self.fname == 'amazon_toy':
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            
            cluster_candidate = torch.index_select(user_id_for_cluster_id, dim=0, index=user_id) # torch.Size([128, 1])
            target_embedding_candidate = torch.index_select(cluster_id_for_finaldim_embedding, dim=0, index=cluster_candidate.squeeze()) # torch.Size([128, 64])
            target_log_feats = target_embedding_candidate.unsqueeze(1).expand(source_log_feats.shape)
            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))
                        
        elif self.fname == 'amazon_game':
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([128, 200, 64])
            cluster_candidate = torch.index_select(user_id_for_cluster_id, dim=0, index=user_id) # torch.Size([128, 1])
            target_embedding_candidate = torch.index_select(cluster_id_for_finaldim_embedding, dim=0, index=cluster_candidate.squeeze()) # torch.Size([128, 64])
            target_log_feats = target_embedding_candidate.unsqueeze(1).expand(source_log_feats.shape)
            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats_time = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats_time)))
            
#         log_feats = log_feats
        log_feats_l2norm = torch.nn.functional.normalize(log_feats, p=2, dim=-1)
        return log_feats_l2norm
    
    def calculate_source_embedding_cluster_all_fortest(self, user_id, source_log_seqs, target_log_seqs, user_id_for_cluster_id, cluster_id_for_finaldim_embedding):
#         ipdb.set_trace()
        if self.fname == 'amazon_toy':
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            
            cluster_candidate = torch.index_select(user_id_for_cluster_id, dim=0, index=user_id) # torch.Size([128, 1])
            target_embedding_candidate = torch.index_select(cluster_id_for_finaldim_embedding, dim=0, index=cluster_candidate.squeeze()) # torch.Size([128, 64])
            
            concatenate_log_feats = torch.cat([source_log_feats[:,-1,:], target_embedding_candidate], dim=-1)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))
                        
        elif self.fname == 'amazon_game':
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([128, 200, 64])
            cluster_candidate = torch.index_select(user_id_for_cluster_id, dim=0, index=user_id) # torch.Size([128, 1])
            target_embedding_candidate = torch.index_select(cluster_id_for_finaldim_embedding, dim=0, index=cluster_candidate.squeeze()) # torch.Size([128, 64])
            
            concatenate_log_feats_time = torch.cat([source_log_feats[:,-1,:], target_embedding_candidate], dim=-1)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats_time)))
            
#         log_feats = log_feats
        log_feats_l2norm = torch.nn.functional.normalize(log_feats, p=2, dim=-1)
        return log_feats_l2norm
    
    
    def calculate_weight_cluster(self, user_id, source_log_seqs, user_train_source_sequence_for_target_indices, neg_seqs_list, pos_target, user_id_for_cluster_id, cluster_id_for_finaldim_embedding):
#         ipdb.set_trace()
        neg_embs = []
        if self.fname == 'amazon_toy':
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            
            cluster_candidate = torch.index_select(user_id_for_cluster_id, dim=0, index=user_id) # torch.Size([128, 1])
            target_embedding_candidate = torch.index_select(cluster_id_for_finaldim_embedding, dim=0, index=cluster_candidate.squeeze()) # torch.Size([128, 64])
            target_log_feats = target_embedding_candidate.unsqueeze(1).expand(source_log_feats.shape)
            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))
                   
            for i in range(0,len(neg_seqs_list)):
                neg_embs.append(self.sasrec_embedding_source.item_emb(neg_seqs_list[i]))
                
        elif self.fname == 'amazon_game':
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([128, 200, 64])
            cluster_candidate = torch.index_select(user_id_for_cluster_id, dim=0, index=user_id) # torch.Size([128, 1])
            target_embedding_candidate = torch.index_select(cluster_id_for_finaldim_embedding, dim=0, index=cluster_candidate.squeeze()) # torch.Size([128, 64])
            target_log_feats = target_embedding_candidate.unsqueeze(1).expand(source_log_feats.shape)
            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats_time = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats_time)))
           
            for i in range(0,len(neg_seqs_list)):
                neg_embs.append(self.sasrec_embedding_target.item_emb(neg_seqs_list[i]))
    
        log_feats = log_feats[torch.where(pos_target!=0)]
        log_feats_l2norm = torch.nn.functional.normalize(log_feats, p=2, dim=-1)
                
        scores_tensor = torch.zeros([len(torch.where(pos_target!=0)[0]),0], dtype=torch.float32, device='cuda')
        for i in range(0,len(neg_seqs_list)):
            neg_embs_i = neg_embs[i][torch.where(pos_target!=0)]
            neg_embs_l2norm_i = torch.nn.functional.normalize(neg_embs_i, p=2, dim=-1)
            neg_logits_i = (log_feats_l2norm * neg_embs_l2norm_i).sum(dim=-1) * self.temperature # torch.Size([128, 200])
            scores_tensor = torch.cat([scores_tensor,neg_logits_i.unsqueeze(-1)], dim=-1)
            
        if self.similar_for_big == 'True_Exp':
            weight = torch.exp(scores_tensor)*self.source_weight
        elif self.similar_for_big == 'True_Softmax_Exp':
            weight = torch.exp(torch.nn.Softmax(dim=1)(scores_tensor))*self.source_weight
        elif self.similar_for_big == 'False_Exp':
            weight = torch.exp(-scores_tensor)*self.source_weight
        elif self.similar_for_big == 'False_Softmax_Exp':
            weight = torch.exp(torch.nn.Softmax(dim=1)(-scores_tensor))*self.source_weight
        weight_list = []
        for i in range(0,len(neg_seqs_list)):
            weight_list.append(weight[:,i])

        return weight_list
    
    
    
    def calculate_weight_cluster_sample(self, user_id, source_log_seqs, user_train_source_sequence_for_target_indices, item_samples, pos_target, user_id_for_cluster_id, cluster_id_for_finaldim_embedding):
#         ipdb.set_trace()
        if self.fname == 'amazon_toy':
            source_log_feats = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            
            cluster_candidate = torch.index_select(user_id_for_cluster_id, dim=0, index=user_id) # torch.Size([128, 1])
            target_embedding_candidate = torch.index_select(cluster_id_for_finaldim_embedding, dim=0, index=cluster_candidate.squeeze()) # torch.Size([128, 64])
            target_log_feats = target_embedding_candidate.unsqueeze(1).expand(source_log_feats.shape)
            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))
                   
            item_samples_embedding = self.sasrec_embedding_source.item_emb(item_samples)

                
        elif self.fname == 'amazon_game':
            source_log_feats = self.sasrec_embedding_source(source_log_seqs) # torch.Size([128, 200, 64])
            cluster_candidate = torch.index_select(user_id_for_cluster_id, dim=0, index=user_id) # torch.Size([128, 1])
            target_embedding_candidate = torch.index_select(cluster_id_for_finaldim_embedding, dim=0, index=cluster_candidate.squeeze()) # torch.Size([128, 64])
            target_log_feats = target_embedding_candidate.unsqueeze(1).expand(source_log_feats.shape)
            
            source_log_feats_time = source_log_feats[torch.tile(torch.arange(0,source_log_seqs.shape[0]).unsqueeze(1), [1, source_log_seqs.shape[1]]).cuda(), user_train_source_sequence_for_target_indices.type(torch.long),:]
            concatenate_log_feats_time = torch.cat([source_log_feats_time, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats_time)))
           
            item_samples_embedding = self.sasrec_embedding_target.item_emb(item_samples)
    
        log_feats = log_feats[torch.where(pos_target!=0)]
        log_feats_l2norm = torch.nn.functional.normalize(log_feats, p=2, dim=-1)
        
        item_samples_embedding_l2norm = torch.nn.functional.normalize(item_samples_embedding, p=2, dim=-1)
        scores = (log_feats_l2norm.unsqueeze(1).expand(item_samples_embedding_l2norm.shape) * item_samples_embedding_l2norm).sum(dim=-1) * self.temperature
#         ipdb.set_trace()
        scores_tensor = torch.where(item_samples != 0, scores, torch.ones_like(scores) * (-2 ** 31))
        if self.similar_for_big == 'True_Exp':
            weight = torch.exp(scores_tensor)*self.source_weight
        elif self.similar_for_big == 'True_Softmax_Exp':
            weight = torch.exp(torch.nn.Softmax(dim=1)(scores_tensor))*self.source_weight
        elif self.similar_for_big == 'False_Exp':
            weight = torch.exp(-scores_tensor)*self.source_weight
        elif self.similar_for_big == 'False_Softmax_Exp':
            weight = torch.exp(torch.nn.Softmax(dim=1)(-scores_tensor))*self.source_weight
        
        return weight