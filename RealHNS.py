import os
import time
import torch
import argparse
import ipdb
import math
from model import SASRec_V2_Adaptive
# from model import SASRec_V1
import pretty_errors
from model import EarlyStopping_onetower
from torch.nn.utils.rnn import pad_sequence
from kmeans_pytorch import kmeans

from utils import *
import os
import io
import copy
from matplotlib.pyplot import MultipleLocator

import matplotlib.pyplot as plt
plt.switch_backend('agg')

# -*- coding: UTF-8 -*-
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=2000)

from matplotlib.font_manager import FontManager
fm = FontManager()
mat_fonts = set(f.name for f in fm.ttflist)
print(mat_fonts)


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

    
# load weights
def get_updateModel(model, path_source, path_target):
#     # load the model
#     pretrained_dict_source = torch.load(path_source, map_location='cpu') # 58
#     pretrained_dict_target = torch.load(path_target, map_location='cpu') # 58
    
#     model_dict = model.state_dict() # 58
#     shared_dict_source = {k: v for k, v in pretrained_dict_source.items() if k.startswith('sasrec_embedding_source')}
#     shared_dict_target = {k: v for k, v in pretrained_dict_target.items() if k.startswith('sasrec_embedding_target')}

#     model_dict.update(shared_dict_source)
#     model_dict.update(shared_dict_target)
    
#     model.load_state_dict(model_dict)
    
#     ipdb.set_trace()
    pretrained_dict_source = torch.load(path_source, map_location='cpu') # 68
    pretrained_dict_target = torch.load(path_target, map_location='cpu') # 68
    model_dict = model.state_dict() # 68
    
    shared_dict_source = {k: v for k, v in pretrained_dict_source.items() if k.startswith('sasrec_embedding_source')}# 28
    shared_dict_target = {k: v for k, v in pretrained_dict_target.items() if k.startswith('sasrec_embedding_target')}# 28

    model_dict.update(shared_dict_source)
    model_dict.update(shared_dict_target)
    print("Load the length of source is:", len(shared_dict_source.keys()))
    print("Load the length of target is:", len(shared_dict_target.keys()))

    model.load_state_dict(model_dict)
    return model

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)


def get_curriculum_DNS_num(epoch, curriculum_start_epoch, DNS_num, source_DNS_num, curriculum_range, curriculum_multiple, curriculum_ratio, average_score_item, curriculum_start_ratio):
    if args.SD_first == 0:
        if epoch <= curriculum_start_epoch:
            curriculum_DNS_num = 0
            curriculum_source_DNS_num = 0
            curriculum_distance = 0
        elif epoch > curriculum_start_epoch:
            curriculum_DNS_all_num = min(math.ceil((epoch-curriculum_start_epoch)/curriculum_range)*curriculum_multiple, (DNS_num+source_DNS_num))
            curriculum_DNS_num = math.ceil(curriculum_DNS_all_num*DNS_num/(DNS_num+source_DNS_num))
            curriculum_source_DNS_num = curriculum_DNS_all_num - curriculum_DNS_num
            curriculum_distance = min((average_score_item/curriculum_start_ratio) * pow(curriculum_ratio, math.ceil((epoch-curriculum_start_epoch)/curriculum_range)), average_score_item)
    elif args.SD_first == 1:
        if epoch <= curriculum_start_epoch:
            curriculum_DNS_num = 0
            curriculum_source_DNS_num = 0
            curriculum_distance = 0
        elif epoch > curriculum_start_epoch:
            curriculum_DNS_all_num = min(math.ceil((epoch-curriculum_start_epoch)/curriculum_range)*curriculum_multiple, (DNS_num+source_DNS_num))
            curriculum_DNS_num = min(curriculum_DNS_all_num, DNS_num)
            curriculum_source_DNS_num = curriculum_DNS_all_num - curriculum_DNS_num
            curriculum_distance = min((average_score_item/curriculum_start_ratio) * pow(curriculum_ratio, math.ceil((epoch-curriculum_start_epoch)/curriculum_range)), average_score_item)
    print("In epoch {}, the curriculum_DNS_num is {}, the curriculum_source_DNS_num is {}, the curriculum_distance is {}".format(epoch, curriculum_DNS_num, curriculum_source_DNS_num, curriculum_distance))
    return curriculum_DNS_num, curriculum_source_DNS_num, curriculum_distance


def get_exclusive(t1, t2):
    
    t1_exclusive = t1[(t1.view(1, -1) != t2.view(-1, 1)).all(dim=0)]
    
    return t1_exclusive


def calculate_distance(feat1,feat2,args):
    if args.distance_function == 'l2_distance':
#         final_distance = torch.dist(feat1, feat2, p=2) # 
        final_distance = pdist_l2(feat1, feat2) # torch.Size([1015])----[pos_target]
    elif args.distance_function == 'hadamard_product':
        final_distance = (feat1*feat2).sum(dim=-1) * args.temperature # 
    return final_distance



def generate_user_sample(usernum,user_train_target):
    cold_user_list = {}
    max_len = 0
#     ipdb.set_trace()
    for k,v in user_train_target.items():
        max_len = max(max_len,len(v))
    
    for i in range(0,max_len+1):
        cold_user_list.update({i:[]})
    
    for k,v in user_train_target.items():
        item_list = cold_user_list[len(v)]
        item_list.append(k)
        cold_user_list.update({len(v):item_list})
#     ipdb.set_trace()
            
    return cold_user_list
    
def generate_user_sample_cluster(epoch, user_list, cluster_id_for_user_index_dict, source_finaldim_embedding, target_finaldim_embedding, cluster_id_for_finaldim_embedding, outlier_weight):
    cluster_user_dict = {}    
    number = 0
    cluster_user_list = []
    for i in range(0, args.num_user_clusters):
        user_id = torch.tensor(np.array(cluster_id_for_user_index_dict[i]), device='cuda')
        finaldim_embedding_target = torch.index_select(target_finaldim_embedding, dim=0, index=user_id) # torch.Size([288, 64])----[pos_targets, hidden_unit]
        score_target = calculate_distance(finaldim_embedding_target.cpu(), cluster_id_for_finaldim_embedding[0].cpu().unsqueeze(0).expand(finaldim_embedding_target.shape), args)
        sort_target_for_indices = torch.sort(input=score_target, dim=0, descending=False, stable=True)[1] # torch.Size([4966])
        sort_target_for_values = torch.sort(input=score_target, dim=0, descending=False, stable=True)[0] # torch.Size([4966])
        sort_target_ids_choose_list = np.array(sort_target_for_indices[:(int(len(user_id)*outlier_weight))]).tolist()

        user_target_ids_choose_list = np.array(torch.tensor(user_list)[user_id[sort_target_ids_choose_list].cpu()]).tolist()

        cluster_user_list.extend(user_target_ids_choose_list)
        number = number + len(user_target_ids_choose_list)
    cluster_user_dict.update({outlier_weight:cluster_user_list})
    print("The sampled user number of weight {} in epoch {} in {} dataset is {}:".format(outlier_weight, epoch, args.dataset, number))
    with io.open(result_path+'/log_file_cluster.txt', 'a', encoding='utf-8') as file:
        file.write("The sampled user number of weight {} in epoch {} in {} is {}\n:".format(outlier_weight, epoch, args.dataset, number))
#     ipdb.set_trace()
    with open(file_result_path+'/cluster_user_dict_'+str(args.dataset)+'_epoch'+str(epoch)+'.pkl', 'wb') as f:
        pkl.dump(cluster_user_dict, f)   
    with open(file_result_path+'/cluster_id_for_user_index_dict_'+str(args.dataset)+'_epoch'+str(epoch)+'.pkl', 'wb') as f:
        pkl.dump(cluster_id_for_user_index_dict, f)   
    torch.save(user_list, file_result_path+'user_list_epoch'+str(args.epoch)+'.pt')
    torch.save(source_finaldim_embedding, file_result_path+'source_finaldim_embedding_epoch'+str(args.epoch)+'.pt')
    torch.save(target_finaldim_embedding, file_result_path+'target_finaldim_embedding_epoch'+str(args.epoch)+'.pt')
    torch.save(cluster_id_for_finaldim_embedding, file_result_path+'cluster_id_for_finaldim_embedding_epoch'+str(args.epoch)+'.pt')
    
    return torch.tensor(cluster_user_list, device='cuda')





parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--cross_dataset', default='Toy_Game')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--num_samples', default=100, type=int)
parser.add_argument('--decay', default=4, type=int)
parser.add_argument('--lr_decay_rate', default=0.99, type=float)
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--version', default='SASRec_V13', type=str)
parser.add_argument('--lr_linear', default=0.01, type=float)
parser.add_argument('--start_decay_linear', default=8, type=int)
parser.add_argument('--temperature', default=5, type=float)
parser.add_argument('--seed', default=3, type=int)
parser.add_argument('--lrscheduler', default='ExponentialLR', type=str)
parser.add_argument('--patience', default=10, type=int)
parser.add_argument('--sample_ratio', default=20, type=int)
parser.add_argument('--negative_sample', default='DNS', type=str)
parser.add_argument('--PNS_num', default=0, type=int)
parser.add_argument('--DNS_num', default=8, type=int)
parser.add_argument('--candidate_num', default=100, type=int)
parser.add_argument('--random_range_min', default=0, type=int)
parser.add_argument('--random_range_max', default=100, type=int)
parser.add_argument('--nearest_num', default=1000, type=int)
parser.add_argument('--candidate_min', default=0, type=int)
parser.add_argument('--candidate_max', default=1000, type=int)
parser.add_argument('--curriculum_start_epoch', default=5, type=int)
parser.add_argument('--curriculum_range', default=2, type=int)
parser.add_argument('--curriculum_multiple', default=1, type=int)
parser.add_argument('--curriculum_ratio', default=1.15, type=float)
parser.add_argument('--curriculum_start_ratio', default=5, type=int)

parser.add_argument('--importance_weight', default=1, type=float)
parser.add_argument('--candidate_min_percentage_user', default=0, type=int)
parser.add_argument('--candidate_max_percentage_user', default=10, type=int)
parser.add_argument('--item_distance_percentage', default=10, type=int)
parser.add_argument('--user_item_distance_percentage', default=10, type=int)

parser.add_argument('--candidate_min_percentage_source', default=0, type=int)
parser.add_argument('--candidate_max_percentage_source', default=30, type=int)
parser.add_argument('--source_DNS_num', default=2, type=int)
parser.add_argument('--random_range_min_source', default=0, type=int)
parser.add_argument('--random_range_max_source', default=100, type=int)

parser.add_argument('--num_user_clusters', default=20, type=int)

parser.add_argument('--distance_function', default='hadamard_product', type=str)
parser.add_argument('--source_weight', default=1.0, type=float)
parser.add_argument('--similar_for_big', default='True', type=str)
parser.add_argument('--result_path', default='None', type=str)
parser.add_argument('--epoch', default=0, type=int)
parser.add_argument('--interval', default=0, type=int)
parser.add_argument('--num_clusters', default=100, type=int)

parser.add_argument('--SD_first', default=1, type=int)
parser.add_argument('--outlier_weight', default=0.005, type=float)

args = parser.parse_args()


SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False



result_path = './results_mk/' + str(args.cross_dataset) + '_100samples/RealHNS/' + str(args.dataset) + '/' + str(args.index) + 'th_seed' + str(args.seed) + '_lr' + str(args.lr) + '_NSas' + str(args.negative_sample) + '_sampleratio' + str(args.sample_ratio) + '_random_range_min' + str(args.random_range_min) + '_random_range_max' + str(args.random_range_max) + '_patience' + str(args.patience) + '/'

print("Save in path:", result_path)
file_result_path = result_path+'save_file/'
if not os.path.isdir(result_path):
    os.makedirs(result_path)
if not os.path.isdir(file_result_path):
    os.makedirs(file_result_path)
with open(os.path.join(result_path, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
# f.close()

args.result_path = result_path


if args.cross_dataset == 'Book_Movie':
    source_name = 'book'
    target_name = 'movie'
elif args.cross_dataset == 'Toy_Game':
    source_name = 'toy'
    target_name = 'game'


if __name__ == '__main__':
    # global dataset
#     ipdb.set_trace()
#     print(os.getcwd())
    dataset = data_partition(args.version, args.dataset, args.cross_dataset, args.maxlen)

#     [user_train_source, user_train_target, user_valid_target, user_test_target, usernum, itemnum, interval] = dataset
    [user_train_mix, user_train_source, user_train_target, user_valid_target, user_test_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, usernum, itemnum, interval] = dataset
#     [user_train_source, user_train_target, user_valid_source, user_valid_target, user_test_source, user_test_target, usernum, itemnum, interval] = dataset
#     num_batch = len(user_train_source) // args.batch_size # 908
    cc_source = 0.0
    cc_target = 0.0
    for u in user_train_source:
        cc_source = cc_source + len(user_train_source[u])
        cc_target = cc_target + len(user_train_target[u])
    print('average sequence length in source domain: %.2f' % (cc_source / len(user_train_source)))
    print('average sequence length in target domain: %.2f' % (cc_target / len(user_train_source)))
    print('average sequence length in both domain: %.2f' % ((cc_source + cc_target) / len(user_train_source)))

    user_list = []
    for u_i in range(1, usernum):
        if len(user_train_source[u_i]) >= 1 and len(user_train_target[u_i]) >= 2: 
            user_list.append(u_i)
    
#     user_list = np.arange(1, usernum + 1)
    np.random.shuffle(user_list)
#     user_list = np.arange(1, usernum)
    num_batch = math.ceil(len(user_list) / args.batch_size) # 908

    sampler = WarpSampler_NS_final(args.version, args.dataset, args.cross_dataset, interval, user_train_mix, user_train_source, user_train_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, user_list, itemnum, None, None, args.sample_ratio, None, None, None, args, num_batch*args.batch_size, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=1)
    model = SASRec_V2_Adaptive(usernum, itemnum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    if args.cross_dataset == 'Toy_Game':
        toy_model_path = './Checkpoint/checkpoint_toy.pt'
        game_model_path = './Checkpoint/checkpoint_game.pt'
        get_updateModel(model, toy_model_path, game_model_path)


    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    pdist_l2 = torch.nn.PairwiseDistance(p=2)
    
    # set the early stop
    early_stopping = EarlyStopping_onetower(args.patience, version='SASRec_V3', verbose=True)  # 关于 EarlyStopping 的代码可先看博客后面的内容

    # set the learning rate scheduler
    if args.lrscheduler == 'Steplr': # 
        learningrate_scheduler = torch.optim.lr_scheduler.StepLR(adam_optimizer, step_size=args.decay, gamma=args.lr_decay_rate, verbose=True)
    elif args.lrscheduler == 'ExponentialLR': # 
        learningrate_scheduler = torch.optim.lr_scheduler.ExponentialLR(adam_optimizer, gamma=args.lr_decay_rate, last_epoch=-1, verbose=True)
    elif args.lrscheduler == 'CosineAnnealingLR':
        learningrate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam_optimizer, T_max=args.num_epochs, eta_min=0, last_epoch=-1, verbose=True)
    
    T = 0.0
    t0 = time.time()
#     ipdb.set_trace()
    epoch_list = []
    lr_list = []
    loss_train_list = []
    loss_test_list = []
    ndcg_list = []
    hr_list = []
    auc_list = []    
    model_score_pos = []
    model_score_neg = []
    model_indice_number = []
    
    if args.dataset=='amazon_toy' or args.dataset=='amazon_book':
        item_number = interval
        candidate_min_user = math.floor(interval * args.candidate_min_percentage_user / 100)
        candidate_max_user = math.ceil(interval * args.candidate_max_percentage_user / 100)
        candidate_min_source = math.floor(interval * args.candidate_min_percentage_source / 100)
        candidate_max_source = math.ceil(interval * args.candidate_max_percentage_source / 100)
    elif args.dataset=='amazon_game' or args.dataset=='amazon_movie':
        item_number = itemnum-interval
        candidate_min_user = math.floor((itemnum-interval) * args.candidate_min_percentage_user / 100)
        candidate_max_user = math.ceil((itemnum-interval) * args.candidate_max_percentage_user / 100)
        candidate_min_source = math.floor((itemnum-interval) * args.candidate_min_percentage_source / 100)
        candidate_max_source = math.ceil((itemnum-interval) * args.candidate_max_percentage_source / 100)

    print("The item_number is:",item_number)
    print("The candidate_min_user is:",candidate_min_user)
    print("The candidate_max_user is:",candidate_max_user)
    print("The candidate_min_source is:",candidate_min_source)
    print("The candidate_max_source is:",candidate_max_source)

#     cold_user_list = generate_user_sample(usernum,user_train_target)
    for epoch in range(1, args.num_epochs + 1):
        args.epoch = epoch
        with_pretrain_score_sum = []
        with_train_score_sum = []
        with_train_score_num = []
        sample_pretrain_score_sum = []
        sample_train_score_sum = []
        model_score_pos_sum = []
        model_score_neg_DNS_sum = []
        model_score_neg_DNS_source_sum = []
        model_score_neg_RNS_sum = []
        pos_order_in_candidate = []
        pos_order_in_range = []
        pos_order_in_sample = []
        raw_DNS_num = []
        raw_DNS_source_num = []
        raw_outlier_source_num = []
        raw_outlier_positive_num = []
        
        distance_user_rep_with_item_rep = []
        distance_user_rep_with_source_rep = []
        distance_random_user_rep = []

        distance_final_user_with_pos_target = []
        distance_source_user_with_pos_target = []
        distance_final_user_with_source_user = []
        
        distance_neg_target_candidate_with_pos_target = []
        distance_neg_source_candidate_with_pos_target = []
#         ipdb.set_trace()
        model.eval()
        with torch.no_grad():
            tt0 = time.time()
            print("Start calculate the user-based similarity...")
            user_nearest_item_raw = torch.zeros([usernum+1, candidate_max_user-candidate_min_user], dtype=torch.int32, device='cuda') # torch.Size([7997, 4961])
            user_nearest_item_score_raw = torch.zeros([usernum+1, candidate_max_user-candidate_min_user], dtype=torch.float32, device='cuda') # torch.Size([7997, 4961])
            # for the source cluster
            source_finaldim_embedding = torch.zeros(size = [0, args.hidden_units], dtype=torch.float32, device='cuda')
            target_finaldim_embedding = torch.zeros(size = [0, args.hidden_units], dtype=torch.float32, device='cuda')            
            if args.dataset=='amazon_toy' or args.dataset=='amazon_book' or args.dataset=='douban_movie':
                item_list = torch.arange(start=1,end=interval+1, step=1, device='cuda', requires_grad=False) # torch.Size([37868])
                item_list_embedding = model.sasrec_embedding_source.item_emb(item_list).cpu() # torch.Size([37868, 64])
                item_list_embedding = torch.nn.functional.normalize(item_list_embedding, p=2, dim=-1)
            elif args.dataset=='amazon_game' or args.dataset=='amazon_movie' or args.dataset=='douban_music':
                item_list = torch.arange(start=interval+1, end=itemnum+1, step=1, device='cuda', requires_grad=False) # torch.Size([11735])
                item_list_embedding = model.sasrec_embedding_target.item_emb(item_list).cpu() # torch.Size([11735, 64])
                item_list_embedding = torch.nn.functional.normalize(item_list_embedding, p=2, dim=-1)

            seq_source_dict = {}
            seq_target_dict = {}
            user_train_source_sequence_for_target_indices_dict = {}
            for u in user_list:
                    # for the movie domain
                seq_source = torch.zeros([1,args.maxlen], dtype=torch.int32, device='cuda') # torch.Size([1, 200])
                seq_target = torch.zeros([1,args.maxlen], dtype=torch.int32, device='cuda') # torch.Size([1, 200])
                user_train_source_sequence_for_target_indices = torch.zeros([1,args.maxlen], dtype=torch.int32, device='cuda') # torch.Size([1, 200])

                idx_source = args.maxlen - 1 #49
                idx_target = args.maxlen - 1 #49

                for i in reversed(range(0, len(user_train_source[u]))): # reversed是逆序搜索，这里的i指的是交互的物品
                    seq_source[0,idx_source] = user_train_source[u][i]
                    # 为什么要设定不等于0？是为了保证当序列长度没到达maxlen时，正样本序列会补充为0，那么构成的负样本序列也应该是0
                    idx_source -= 1
                    if idx_source == -1: break

                for i in reversed(range(0, len(user_train_target[u][:-1]))): # reversed是逆序搜索，这里的i指的是交互的物品
                    seq_target[0,idx_target] = user_train_target[u][i]

                    if user_train_source_sequence_for_target[u][i] < -args.maxlen or user_train_source_sequence_for_target[u][i] == -len(user_train_source[u])-1:
                        user_train_source_sequence_for_target_indices[0,idx_target] = 0
                    else:
                        user_train_source_sequence_for_target_indices[0,idx_target] = user_train_source_sequence_for_target[u][i] + args.maxlen

                    idx_target -= 1
                    if idx_target == -1: break

                seq_source_dict.update({u:seq_source})
                seq_target_dict.update({u:seq_target})
                user_train_source_sequence_for_target_indices_dict.update({u:user_train_source_sequence_for_target_indices})
                scores, source_log_feats_l2norm, target_log_feats_l2norm = model.calculate_score_withembedding(seq_source, seq_target, user_train_source_sequence_for_target_indices, item_list)
                source_finaldim_embedding = torch.cat([source_finaldim_embedding, source_log_feats_l2norm], dim=0)
                target_finaldim_embedding = torch.cat([target_finaldim_embedding, target_log_feats_l2norm], dim=0)

                sort_indices = torch.sort(input=scores, dim=0, descending=True, stable=True)[1][candidate_min_user:candidate_max_user+len(user_train_target[u])+2] # torch.Size([4966])
                sort_values = torch.sort(input=scores, dim=0, descending=True, stable=True)[0][candidate_min_user:candidate_max_user+len(user_train_target[u])+2] # torch.Size([4966])
                user_indices = item_list[sort_indices] # torch.Size([1001])
                user_indices_score = copy.deepcopy(sort_values)
                for it in (user_train_target[u]+user_valid_target[u]+user_test_target[u]):
                    if it in user_indices:
                        user_equal_it_index = torch.nonzero(user_indices == it).squeeze(1)
                        user_indices = del_tensor_ele(user_indices, user_equal_it_index)
                        user_indices_score = del_tensor_ele(user_indices_score, user_equal_it_index)

                user_nearest_item_raw[u] = user_indices[:candidate_max_user-candidate_min_user]
                user_nearest_item_score_raw[u] = user_indices_score[:candidate_max_user-candidate_min_user]
                del seq_source
                del seq_target
                del user_train_source_sequence_for_target_indices
                del scores

            tt1 = time.time()
    #         ipdb.set_trace()
            random_index = torch.randperm(user_nearest_item_raw.shape[1])
            user_nearest_item = copy.deepcopy(user_nearest_item_raw[:, random_index[:args.nearest_num]])
            user_nearest_item_score = copy.deepcopy(user_nearest_item_score_raw[:, random_index[:args.nearest_num]])
            del user_nearest_item_raw
            del user_nearest_item_score_raw
            torch.cuda.empty_cache()

            print("End calculate the user-based similirity, time cost is {:.4f}s:".format(tt1 - tt0)) # 18.1574s

            score_nonzero_num = len(torch.where(user_nearest_item>0)[0])
            score_nonzero_sum = torch.sum(user_nearest_item_score).item()
            print("The mean value of pretrain nearest item score in epoch {} is {:.4f}\n".format(epoch, score_nonzero_sum / score_nonzero_num))
            with io.open(result_path + 'mean_score.txt', 'a', encoding='utf-8') as file:
                file.write("The mean value of pretrain nearest item score in epoch {} is {:.4f}\n".format(epoch, score_nonzero_sum / score_nonzero_num))

    #         ipdb.set_trace()
            print("The cluster user number is:",len(user_list))
            cluster_ids, cluster_centers = kmeans(X=item_list_embedding, num_clusters=args.num_clusters, distance='cosine', tqdm_flag=False, device='cuda')

            cluster_for_item_index_dict = {}
            cluster_for_item_dict = {}
            item_index_for_cluster = -torch.ones(size = item_list.shape, dtype=torch.int32, device='cuda')
            item_for_cluster = -torch.ones(size = [itemnum + 1], dtype=torch.int32, device='cuda')
    #         distance_item_all = torch.empty(size=[0], dtype=torch.float32, device='cuda')
            for i in range(0, args.num_clusters):
                cluster_for_item_index_dict.update({i:[]})
                cluster_for_item_dict.update({i:[]})

                # the relation of cluster_id and user_index 
            for i in range(0, len(item_list)):
                cluster_for_item_index_raw = cluster_for_item_index_dict[cluster_ids[i].item()]
                cluster_for_item_index_raw.append(i)
                cluster_for_item_index_dict.update({cluster_ids[i].item():cluster_for_item_index_raw})

                cluster_for_item_raw = cluster_for_item_dict[cluster_ids[i].item()]
                cluster_for_item_raw.append(item_list[i].item())
                cluster_for_item_dict.update({cluster_ids[i].item():cluster_for_item_raw})

                # the relation of user_id and 
            for i in range(0, len(item_list)):
                item_index_for_cluster[i] = cluster_ids[i].item()
                item_for_cluster[item_list[i]] = cluster_ids[i].item()

            distance_cluster_all = []
            for i in range(0, args.num_clusters):
                item_center_distance = calculate_distance(torch.index_select(item_list_embedding, dim=0, index=torch.tensor(cluster_for_item_index_dict[i])), cluster_centers[i].unsqueeze(0).expand(len(cluster_for_item_index_dict[i]), -1), args) # torch.Size([1433])
    #             distance_item_all = torch.cat([distance_item_all, item_center_distance.cuda()],dim=0)
                distance_cluster_all.append(torch.mean(item_center_distance).item())

    #         average_score_item = torch.mean(distance_item_all)
            average_score_item = np.mean(distance_cluster_all)

            user_cluster_id_for_user_index_dict = {}
            user_id_for_user_cluster_id = -torch.ones(size = [usernum + 1, 1], dtype=torch.int32, device='cuda')
            user_cluster_id_for_finaldim_embedding = torch.zeros(size = [0, args.hidden_units], dtype=torch.float32, device='cuda')

            user_cluster_ids, user_cluster_centers = kmeans(X=source_finaldim_embedding, num_clusters=args.num_user_clusters, distance='cosine', tqdm_flag=False, device='cuda')

            for i in range(0, args.num_user_clusters):
                user_cluster_id_for_user_index_dict.update({i:[]})

                # the relation of cluster_id and user_index 
            for i in range(0, len(user_list)):
                user_cluster_id_for_user_index_raw = user_cluster_id_for_user_index_dict[user_cluster_ids[i].item()]
                user_cluster_id_for_user_index_raw.append(i)
                user_cluster_id_for_user_index_dict.update({user_cluster_ids[i].item():user_cluster_id_for_user_index_raw})

                # the relation of user_id and 
            for i in range(0, len(user_list)):
                user_id_for_user_cluster_id[user_list[i]] = user_cluster_ids[i].item()

    #         ipdb.set_trace()
            for i in range(0, args.num_user_clusters):
                user_cluster_id_for_user_index_raw = torch.tensor(np.array(user_cluster_id_for_user_index_dict[i]), device='cuda')
                finaldim_embedding_target = torch.index_select(target_finaldim_embedding, dim=0, index=user_cluster_id_for_user_index_raw) # torch.Size([242, 64])----[pos_targets, hidden_unit]
                finaldim_embedding_target_mean = torch.mean(finaldim_embedding_target, dim=0)
                user_cluster_id_for_finaldim_embedding = torch.cat([user_cluster_id_for_finaldim_embedding, finaldim_embedding_target_mean.unsqueeze(0)], dim=0)

                if len(torch.index_select(user_cluster_ids.cuda(), dim=0, index=user_cluster_id_for_user_index_raw).unique()) != 1:
                    ipdb.set_trace()

#             generate_user_sample_cluster(epoch, user_list, user_cluster_id_for_user_index_dict, source_finaldim_embedding, target_finaldim_embedding, user_cluster_id_for_finaldim_embedding)
            outlier_user_tensor = generate_user_sample_cluster(epoch, user_list, user_cluster_id_for_user_index_dict, source_finaldim_embedding, target_finaldim_embedding, user_cluster_id_for_finaldim_embedding, args.outlier_weight)

            print("Start calculate the source-based similarity...")
#             ipdb.set_trace()
            tt2 = time.time()
            user_nearest_source_raw = torch.zeros([usernum+1, candidate_max_source-candidate_min_source], dtype=torch.int32, device='cuda') # torch.Size([7997, 3787])
            user_nearest_source_score_raw = torch.zeros([usernum+1, candidate_max_source-candidate_min_source], dtype=torch.float32, device='cuda') # torch.Size([7997, 3787])

            for u in user_list:
#                 ipdb.set_trace()
                scores = model.calculate_score_source(u, seq_source_dict[u], user_train_source_sequence_for_target_indices_dict[u], item_list, user_id_for_user_cluster_id, user_cluster_id_for_finaldim_embedding)
                sort_source_indices = torch.sort(input=scores, dim=0, descending=True, stable=True)[1][candidate_min_source:candidate_max_source+len(user_train_target[u])+2] # torch.Size([4966])
                sort_source_values = torch.sort(input=scores, dim=0, descending=True, stable=True)[0][candidate_min_source:candidate_max_source+len(user_train_target[u])+2] # torch.Size([4966])
                user_source_indices = item_list[sort_source_indices] # torch.Size([1001])
                user_source_indices_score = copy.deepcopy(sort_source_values)

                for it in (user_train_target[u]+user_valid_target[u]+user_test_target[u]):
                    if it in user_source_indices:
                        user_equal_it_index_source = torch.nonzero(user_source_indices == it).squeeze(1)
                        user_source_indices = del_tensor_ele(user_source_indices, user_equal_it_index_source)
                        user_source_indices_score = del_tensor_ele(user_source_indices_score, user_equal_it_index_source)

                user_nearest_source_raw[u] = user_source_indices[:candidate_max_source-candidate_min_source]
                user_nearest_source_score_raw[u] = user_source_indices_score[:candidate_max_source-candidate_min_source]

                del scores

            tt3 = time.time()
            print("End calculate the user-based similirity, time cost is {:.4f}s:".format(tt3 - tt2)) # 18.1574s


            random_index_source = torch.randperm(user_nearest_source_raw.shape[1])
            user_nearest_source = copy.deepcopy(user_nearest_source_raw[:, random_index_source[:args.nearest_num]])
            user_nearest_source_score = copy.deepcopy(user_nearest_source_score_raw[:, random_index_source[:args.nearest_num]])
            
            del seq_source_dict
            del seq_target_dict
            del user_train_source_sequence_for_target_indices_dict
            del user_nearest_source_raw
            del user_nearest_source_score_raw
            del source_finaldim_embedding
            del target_finaldim_embedding
            torch.cuda.empty_cache()

            source_score_nonzero_num = len(torch.where(user_nearest_source>0)[0])
            source_score_nonzero_sum = torch.sum(user_nearest_source_score).item()
            print("The mean value of pretrain nearest source score in epoch {} is {:.4f}\n".format(epoch, source_score_nonzero_sum / source_score_nonzero_num))
            with io.open(result_path + 'mean_score.txt', 'a', encoding='utf-8') as file:
                file.write("The mean value of pretrain nearest source score in epoch {} is {:.4f}\n".format(epoch, source_score_nonzero_sum / source_score_nonzero_num))
            
#         ipdb.set_trace()
        curriculum_DNS_num, curriculum_source_DNS_num, curriculum_score = get_curriculum_DNS_num(epoch, args.curriculum_start_epoch, args.DNS_num, args.source_DNS_num, args.curriculum_range, args.curriculum_multiple, args.curriculum_ratio, average_score_item, args.curriculum_start_ratio)
            
        epoch_list.append(epoch)
        lr_list.append(learningrate_scheduler.get_last_lr())

        t1 = time.time()
        loss_epoch = 0
        model.train()
        u_all_tensor = torch.tensor([], dtype=torch.int32)
#         u_embedding_dict = {}
        score_final_user_with_pos_target = []
        score_neg_target_candidate_with_pos_target = []
        score_random_user_rep = []
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            importance_weight_gt = 0
#             u, seq_source, seq_target, pos_target, neg_target = sampler.next_batch() # tuples to ndarray
            u, seq_mix, seq_source, seq_target, pos_target, neg_target, user_train_mix_sequence_for_target_indices, user_train_source_sequence_for_target_indices = sampler.next_batch() # tuples to ndarray
            u, seq_source, seq_target, pos_target, neg_target, user_train_mix_sequence_for_target_indices, user_train_source_sequence_for_target_indices = np.array(u), np.array(seq_source), np.array(seq_target), np.array(pos_target), np.array(neg_target), np.array(user_train_mix_sequence_for_target_indices), np.array(user_train_source_sequence_for_target_indices)      
            u, seq_source, seq_target, pos_target, neg_target, user_train_mix_sequence_for_target_indices, user_train_source_sequence_for_target_indices = torch.tensor(u).cuda(), torch.tensor(seq_source).cuda(), torch.tensor(seq_target).cuda(), torch.tensor(pos_target).cuda(), torch.tensor(neg_target).cuda(), torch.tensor(user_train_mix_sequence_for_target_indices).cuda(), torch.tensor(user_train_source_sequence_for_target_indices).cuda() 
            
            if step ==0:
                print("user:",u)
            u_all_tensor = torch.cat((u_all_tensor,u.cpu()), 0)
            
            neg_target_list = []
#             ipdb.set_trace()
            with torch.no_grad():
                model.eval()
                if args.dataset=='amazon_toy':
                    ############################################ for single domain #######################################
#                     ipdb.set_trace()
                    final_user_embedding = model.calculate_embedding(seq_source, seq_target, user_train_source_sequence_for_target_indices)[torch.where(pos_target != 0)] # torch.Size([128, 200, 64])
                    pos_target_embedding = model.sasrec_embedding_source.item_emb(pos_target[torch.where(pos_target!=0)]) # torch.Size([1015, 64])----[pos_targets, embedding size]----pos target embedding
                    pos_target_embedding = torch.nn.functional.normalize(pos_target_embedding, p=2, dim=-1)

                    all_score = calculate_distance(final_user_embedding, pos_target_embedding, args) # 

                    neg_target_candidate = torch.index_select(user_nearest_item, dim=0, index=u[torch.where(pos_target!=0)[0]]) # torch.Size([1015, 1000])----[pos_targets, nearest_num]----nearest item of each user
                    neg_target_candidate_embedding = model.sasrec_embedding_source.item_emb(neg_target_candidate)
                    neg_target_candidate_embedding = torch.nn.functional.normalize(neg_target_candidate_embedding, p=2, dim=-1)
                    ############################################ for cross domain #######################################
#                     ipdb.set_trace()
                        # distance between final_user and source_user
                    final_source_embedding = model.calculate_source_embedding_cluster(u, seq_source, seq_target, user_train_source_sequence_for_target_indices, pos_target, user_id_for_user_cluster_id, user_cluster_id_for_finaldim_embedding)
                    all_score_source = calculate_distance(final_user_embedding, final_source_embedding, args)                    
                        # distance between source_user with pos_target 
                    all_score_source_pos = calculate_distance(final_source_embedding, pos_target_embedding, args) 

                    neg_source_candidate = torch.index_select(user_nearest_source, dim=0, index=u[torch.where(pos_target!=0)[0]]) # torch.Size([1015, 1000])----[pos_targets, nearest_num]----nearest item of each user
                    neg_source_candidate_embedding = model.sasrec_embedding_source.item_emb(neg_source_candidate)
                    neg_source_candidate_embedding = torch.nn.functional.normalize(neg_source_candidate_embedding, p=2, dim=-1)
                    
                          # calculate the score
                    items_score = calculate_distance(neg_target_candidate_embedding, pos_target_embedding.unsqueeze(1).expand(-1, neg_target_candidate_embedding.shape[1], -1), args) # torch.Size([1433, 1000])
                    items_source_score = calculate_distance(neg_source_candidate_embedding, pos_target_embedding.unsqueeze(1).expand(-1, neg_source_candidate_embedding.shape[1], -1), args) # torch.Size([1433, 1000])
                    
                    final_candidate = torch.where(items_score <= curriculum_score, neg_target_candidate, torch.zeros(size=neg_target_candidate.shape, dtype=torch.int32, device='cuda')) # Filter out items that are not close to the target item
                    final_candidate_source = torch.where(items_source_score <= curriculum_score, neg_source_candidate, torch.zeros(size=neg_source_candidate.shape, dtype=torch.int32, device='cuda')) # Filter out items that are not close to the target item

                    target_neg_embedding = model.sasrec_embedding_source.item_emb(final_candidate) # torch.Size([128, 100, 200, 64])
                    target_neg_embedding = torch.nn.functional.normalize(target_neg_embedding, p=2, dim=-1)
                    source_neg_embedding = model.sasrec_embedding_source.item_emb(final_candidate_source) 
                    source_neg_embedding = torch.nn.functional.normalize(source_neg_embedding, p=2, dim=-1)
                elif args.dataset=='amazon_game':
                    ############################################ for single domain #######################################
                    final_user_embedding = model.calculate_embedding(seq_source, seq_target, user_train_source_sequence_for_target_indices)[torch.where(pos_target != 0)] # torch.Size([128, 200, 64])----[batch size, maxlen, embedding size]----final user embedding
                    pos_target_embedding = model.sasrec_embedding_target.item_emb(pos_target[torch.where(pos_target!=0)]) # torch.Size([1015, 64])----[pos_targets, embedding size]----pos target embedding
                    pos_target_embedding = torch.nn.functional.normalize(pos_target_embedding, p=2, dim=-1)

                    all_score = calculate_distance(final_user_embedding, pos_target_embedding, args) # 
                    neg_target_candidate = torch.index_select(user_nearest_item, dim=0, index=u[torch.where(pos_target!=0)[0]]) # torch.Size([1015, 1000])----[pos_targets, nearest_num]----nearest item of each user
                    neg_target_candidate_embedding = model.sasrec_embedding_target.item_emb(neg_target_candidate)
                    neg_target_candidate_embedding = torch.nn.functional.normalize(neg_target_candidate_embedding, p=2, dim=-1)
                    ############################################ for single domain #######################################
#                     ipdb.set_trace()
                    final_source_embedding = model.calculate_source_embedding_cluster(u, seq_source, seq_target, user_train_source_sequence_for_target_indices, pos_target, user_id_for_user_cluster_id, user_cluster_id_for_finaldim_embedding)
                        # distance between final_user and source_user
                    all_score_source = calculate_distance(final_user_embedding, final_source_embedding, args) 
                        # distance between source_user with pos_target
                    all_score_source_pos = calculate_distance(final_source_embedding, pos_target_embedding, args) # 
                            
                    neg_source_candidate = torch.index_select(user_nearest_source, dim=0, index=u[torch.where(pos_target!=0)[0]]) # torch.Size([1015, 1000])----[pos_targets, nearest_num]----nearest item of each user
                    neg_source_candidate_embedding = model.sasrec_embedding_target.item_emb(neg_source_candidate)
                    neg_source_candidate_embedding = torch.nn.functional.normalize(neg_source_candidate_embedding, p=2, dim=-1)
                    
                    
                    items_score = calculate_distance(neg_target_candidate_embedding, pos_target_embedding.unsqueeze(1).expand(-1, neg_target_candidate_embedding.shape[1], -1), args) # torch.Size([1433, 1000])
                    items_source_score = calculate_distance(neg_source_candidate_embedding, pos_target_embedding.unsqueeze(1).expand(-1, neg_source_candidate_embedding.shape[1], -1), args) # torch.Size([1433, 1000])

                    final_candidate = torch.where(items_score <= curriculum_score, neg_target_candidate, torch.zeros(size=neg_target_candidate.shape, dtype=torch.int32, device='cuda')) # Filter out items that are not close to the target item                     
                    final_candidate_source = torch.where(items_source_score <= curriculum_score, neg_source_candidate, torch.zeros(size=neg_source_candidate.shape, dtype=torch.int32, device='cuda')) # Filter out items that are not close to the target item                     

                    target_neg_embedding = model.sasrec_embedding_target.item_emb(final_candidate) # torch.Size([128, 100, 200, 64])
                    target_neg_embedding = torch.nn.functional.normalize(target_neg_embedding, p=2, dim=-1)
                    source_neg_embedding = model.sasrec_embedding_target.item_emb(final_candidate_source) 
                    source_neg_embedding = torch.nn.functional.normalize(source_neg_embedding, p=2, dim=-1)
                random_score = calculate_distance(final_user_embedding[0], final_user_embedding[-1], args)
                ############################################ for single domain #######################################
                scores = calculate_distance(final_user_embedding.unsqueeze(1).expand([-1, args.nearest_num, -1]), target_neg_embedding, args)
                sample_indices = torch.sort(input=scores, dim=-1, descending=True, stable=True)[1][:, args.random_range_min: args.random_range_max] # torch.Size([128, 200, 200])
                sample_indices_scores = torch.sort(input=scores, dim=-1, descending=True, stable=True)[0][:, args.random_range_min: args.random_range_max] # torch.Size([128, 200, 200])

                random_indices = torch.randint(low=0, high=args.random_range_max-args.random_range_min, size=[scores.shape[0], curriculum_DNS_num+curriculum_source_DNS_num], device='cuda', requires_grad=False) # torch.Size([1015, 10])
                sample_random_indices = torch.gather(input=sample_indices,dim=-1,index=random_indices) # torch.Size([128, 200, 2])

                neg_target_DNS = torch.gather(input=final_candidate, dim=-1, index=sample_random_indices) # torch.Size([1015, 10])
                neg_target_train_score_DNS = torch.gather(input=scores, dim=-1, index=sample_random_indices)
                ############################################# for cross domain #########################################
#                 ipdb.set_trace()
                scores_source = calculate_distance(final_source_embedding.unsqueeze(1).expand([-1, args.nearest_num, -1]), source_neg_embedding, args) # torch.Size([128, 200, 1000])
                sample_indices_source = torch.sort(input=scores_source, dim=-1, descending=True, stable=True)[1][:, args.random_range_min_source: args.random_range_max_source] # torch.Size([128, 200, 200])
                sample_indices_scores_source = torch.sort(input=scores_source, dim=-1, descending=True, stable=True)[0][:, args.random_range_min_source: args.random_range_max_source] # torch.Size([128, 200, 200])

                random_indices_source = torch.randint(low=0, high=args.random_range_max_source-args.random_range_min_source, size=[scores_source.shape[0], curriculum_source_DNS_num], device='cuda', requires_grad=False) # torch.Size([1015, 10])
                sample_random_indices_source = torch.gather(input=sample_indices_source,dim=-1,index=random_indices_source) # torch.Size([128, 200, 2])

                neg_target_DNS_source = torch.gather(input=final_candidate_source, dim=-1, index=sample_random_indices_source)
                neg_target_train_score_DNS_source = torch.gather(input=scores_source, dim=-1, index=sample_random_indices_source)
                
                distance_final_user_with_pos_target.append(all_score.mean().item())
                distance_source_user_with_pos_target.append(all_score_source_pos.mean().item())
                distance_final_user_with_source_user.append(all_score_source.mean().item())
               
                distance_neg_target_candidate_with_pos_target.append(items_score.mean().item())
                distance_neg_source_candidate_with_pos_target.append(items_source_score.mean().item())
                distance_random_user_rep.append(random_score.item())
                if args.dataset=='amazon_toy':
                    pos_order_in_candidate.append(len(torch.where(scores > (final_user_embedding * torch.nn.functional.normalize(model.sasrec_embedding_source.item_emb(pos_target[torch.where(pos_target != 0)]), p=2, dim=-1)).sum(dim=-1).unsqueeze(1).expand([-1,neg_target_candidate.shape[1]]))[0]) / len(torch.where(pos_target != 0)[0]))
                    pos_order_in_range.append(len(torch.where(sample_indices_scores > (final_user_embedding * torch.nn.functional.normalize(model.sasrec_embedding_source.item_emb(pos_target[torch.where(pos_target != 0)]), p=2, dim=-1)).sum(dim=-1).unsqueeze(1).expand([-1,args.random_range_max-args.random_range_min]))[0])/len(torch.where(pos_target!=0)[0]))
                    pos_order_in_sample.append(len(torch.where(neg_target_train_score_DNS > (final_user_embedding * torch.nn.functional.normalize(model.sasrec_embedding_source.item_emb(pos_target[torch.where(pos_target != 0)]), p=2, dim=-1)).sum(dim=-1).unsqueeze(1).expand([-1,curriculum_DNS_num+curriculum_source_DNS_num]))[0]) / len(torch.where(pos_target != 0)[0]))
                elif args.dataset=='amazon_game':                    
                    pos_order_in_candidate.append(len(torch.where(scores > (final_user_embedding * torch.nn.functional.normalize(model.sasrec_embedding_target.item_emb(pos_target[torch.where(pos_target != 0)]), p=2, dim=-1)).sum(dim=-1).unsqueeze(1).expand([-1,neg_target_candidate.shape[1]]))[0]) / len(torch.where(pos_target != 0)[0]))
                    pos_order_in_range.append(len(torch.where(sample_indices_scores > (final_user_embedding * torch.nn.functional.normalize(model.sasrec_embedding_target.item_emb(pos_target[torch.where(pos_target != 0)]), p=2, dim=-1)).sum(dim=-1).unsqueeze(1).expand([-1,args.random_range_max-args.random_range_min]))[0])/len(torch.where(pos_target!=0)[0]))
                    pos_order_in_sample.append(len(torch.where(neg_target_train_score_DNS > (final_user_embedding * torch.nn.functional.normalize(model.sasrec_embedding_target.item_emb(pos_target[torch.where(pos_target != 0)]), p=2, dim=-1)).sum(dim=-1).unsqueeze(1).expand([-1,curriculum_DNS_num+curriculum_source_DNS_num]))[0]) / len(torch.where(pos_target != 0)[0]))
          
                score_final_user_with_pos_target.append(all_score.mean().item())
                score_neg_target_candidate_with_pos_target.append(items_score.mean().item())
                score_random_user_rep.append(random_score.item())
                
                del final_user_embedding
                del final_source_embedding
                del pos_target_embedding
                del all_score
                del all_score_source
                del neg_target_candidate
                del neg_source_candidate
                del neg_target_candidate_embedding
                del neg_source_candidate_embedding
                del target_neg_embedding
                del source_neg_embedding
                del final_candidate
                del final_candidate_source
                del scores
                del scores_source
#                 torch.cuda.empty_cache()
#             ipdb.set_trace()
            mask = torch.sum(u.unsqueeze(1) == outlier_user_tensor, dim=1).unsqueeze(1).expand(args.batch_size, args.maxlen)
            if step == 0:
                single_dns_num = 0
                cross_dns_num = 0
                rns_num = 0
                outlier_num = 0
            outlier_num = outlier_num + len(torch.where(mask>0)[0].unique())
            if epoch > args.curriculum_start_epoch:
                for k in range(0, args.sample_ratio):
                    if k < curriculum_DNS_num:
                        neg_DNS = torch.zeros(size=[args.batch_size, args.maxlen], dtype=torch.int32, device='cuda')
                        neg_DNS[torch.where(pos_target != 0)] = neg_target_DNS[:,k]
                        raw_DNS_num.append(len(torch.where(neg_DNS != 0)[0]))
                        neg_DNS = torch.where(neg_DNS != 0, neg_DNS, neg_target[:,k, :])
                        neg_target_list.append(neg_DNS)
                        if step == 0:
                            single_dns_num = single_dns_num + 1
                    elif k < curriculum_DNS_num+curriculum_source_DNS_num:
                        neg_DNS_single = torch.zeros(size=[args.batch_size, args.maxlen], dtype=torch.int32, device='cuda')
                        neg_DNS_cross = torch.zeros(size=[args.batch_size, args.maxlen], dtype=torch.int32, device='cuda')
                        neg_DNS_single[torch.where(pos_target != 0)] = neg_target_DNS[:,k]
                        neg_DNS_cross[torch.where(pos_target != 0)] = neg_target_DNS_source[:,k-curriculum_DNS_num]
                        neg_DNS = torch.where(mask == 1, neg_DNS_cross, neg_DNS_single)
                        raw_DNS_source_num.append(len(torch.where(neg_DNS != 0)[0]))
                        raw_outlier_source_num.append(len(torch.where(neg_DNS[torch.where(mask>0)[0].unique()]!= 0)[0]))
                        raw_outlier_positive_num.append(len(torch.where(pos_target[torch.where(mask>0)[0].unique()]!= 0)[0]))
                        neg_DNS = torch.where(neg_DNS != 0, neg_DNS, neg_target[:,k, :])
                        neg_target_list.append(neg_DNS)
                        
                        if step == 0:
                            cross_dns_num = cross_dns_num + 1
                    else:
                        neg_target_list.append(neg_target[:,k,:])
                        if step == 0:
                            rns_num = rns_num + 1
            else:
                for k in range(0, args.sample_ratio):
                    neg_target_list.append(neg_target[:,k,:])
                    if step == 0:
                        rns_num = rns_num + 1
            if step == 0:
                with io.open(result_path + 'distance_record.txt', 'a', encoding='utf-8') as file:
                    file.write("Epoch {}:\n".format(epoch)) 
                    file.write("    The num of single-DNS samples is {}, the num of cross-DNS samples is {}, and the num of RNS samples is {:.4f}\n".format(single_dns_num, cross_dns_num, rns_num)) 
            model.train()
            pos_logits, neg_logits = model(u, seq_source, seq_target, pos_target, neg_target_list, user_train_source_sequence_for_target_indices)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(pos_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            score_pos = torch.tensor(0.0)
            indices = torch.where(pos_target != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            score_pos = torch.sum(pos_logits[indices]).detach().cpu()
            score_DNS = torch.tensor(0.0)
            score_DNS_source = torch.tensor(0.0)
            score_RNS = torch.tensor(0.0)
            for k in range(0, args.sample_ratio):
                if k < curriculum_DNS_num:
                    loss += bce_criterion(neg_logits[k][indices], neg_labels[indices]) / args.sample_ratio
                    score_DNS += torch.sum(neg_logits[k][indices]).detach().cpu() / curriculum_DNS_num
                elif k < curriculum_DNS_num + curriculum_source_DNS_num:
                    loss += bce_criterion(neg_logits[k][indices], neg_labels[indices]) / args.sample_ratio
                    score_DNS_source += torch.sum(neg_logits[k][indices]).detach().cpu() / curriculum_source_DNS_num
                else:
                    loss += bce_criterion(neg_logits[k][indices], neg_labels[indices]) / args.sample_ratio
                    score_RNS += torch.sum(neg_logits[k][indices]).detach().cpu() / (args.sample_ratio-curriculum_DNS_num)

            model_score_pos_sum.append(score_pos.item())
            model_score_neg_DNS_sum.append(score_DNS.item())  
            model_score_neg_DNS_source_sum.append(score_DNS_source.item())
            model_score_neg_RNS_sum.append(score_RNS.item())
            with_train_score_num.append(len(indices[0]))

            if epoch == 1 and step == 0:
                with io.open(result_path + 'test_performance.txt', 'a', encoding='utf-8') as file:
                    file.write('Init batch in epoch 1: pos_score is %.4f, neg_DNS_score is %.4f, neg_DNS_source_score is %.4f, neg_RNS_score is %.4f\n' % (sum(model_score_pos_sum) / sum(with_train_score_num), sum(model_score_neg_DNS_sum) / sum(with_train_score_num), sum(model_score_neg_DNS_source_sum) / sum(with_train_score_num), sum(model_score_neg_RNS_sum) / sum(with_train_score_num)))
        
            loss_epoch += loss.item()
            
            loss.backward()
            adam_optimizer.step()
#             ipdb.set_trace()
            if curriculum_DNS_num == 0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) 
                with io.open(result_path + 'loss_log.txt', 'a', encoding='utf-8') as file:
                    file.write("loss in epoch {} iteration {}: {}\n".format(epoch, step, loss.item()))   
            elif curriculum_source_DNS_num == 0:
                print("In epoch {} iteration {}: loss is {}, SDNS_num is {}, Pos_num is {}.".format(epoch, step, loss.item(), raw_DNS_num[-1], with_train_score_num[-1])) 
                print("                          SDNS_prob is {}".format(raw_DNS_num[-1]/with_train_score_num[-1])) 
                with io.open(result_path + 'loss_log.txt', 'a', encoding='utf-8') as file:
                    file.write("In epoch {} iteration {}: loss is {}, SDNS_num is {}, Pos_num is {}\n.".format(epoch, step, loss.item(), raw_DNS_num[-1], with_train_score_num[-1]))  
                    file.write(".   SDNS_prob is {}\n".format(raw_DNS_num[-1]/with_train_score_num[-1]))
            else:
                print("In epoch {} iteration {}: loss is {}, SDNS_num is {}, CDNS_num is {}, Pos_num is {}.".format(epoch, step, loss.item(), raw_DNS_num[-1], raw_DNS_source_num[-1], with_train_score_num[-1])) 
                print("                          SDNS_prob is {}, CDNS_prob is {}.".format(raw_DNS_num[-1]/with_train_score_num[-1], raw_DNS_source_num[-1]/with_train_score_num[-1])) 

                with io.open(result_path + 'loss_log.txt', 'a', encoding='utf-8') as file:
                    file.write("In epoch {} iteration {}: loss is {}, SDNS_num is {}, CDNS_num is {}, Pos_num is {}.\n".format(epoch, step, loss.item(), raw_DNS_num[-1], raw_DNS_source_num[-1], with_train_score_num[-1]))  
                    file.write("    SDNS_prob is {}, CDNS_prob is {}.\n".format(raw_DNS_num[-1]/with_train_score_num[-1], raw_DNS_source_num[-1]/with_train_score_num[-1]))


        loss_train_list.append(loss_epoch / num_batch)
        learningrate_scheduler.step()
        t2 = time.time()
#         ipdb.set_trace()
        with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
            file.write("loss in epoch {}: {}, time: {}\n".format(epoch, loss_epoch / num_batch, t2 - t1))
        with io.open(result_path + 'distance_record.txt', 'a', encoding='utf-8') as file:
            file.write("Epoch {}:\n".format(epoch))
            file.write("    The curriculum_DNS_num is {}, the curriculum_source_DNS_num is {}, the curriculum_distance is {}\n".format(curriculum_DNS_num, curriculum_source_DNS_num, curriculum_score))
            file.write("    The average score is {:.4f}\n".format(average_score_item))      
            file.write("    The score between final_user and pos_target is {:.4f}\n".format(np.mean(distance_final_user_with_pos_target))) 
            file.write("    The score between source_user and pos_target is {:.4f}\n".format(np.mean(distance_source_user_with_pos_target))) 
            file.write("    The score between neg_target_candidate and pos_target is {:.4f}\n".format(np.mean(distance_neg_target_candidate_with_pos_target))) 
            file.write("    The score between neg_source_candidate and pos_target is {:.4f}\n".format(np.mean(distance_neg_source_candidate_with_pos_target))) 
            file.write("    The score between final_user and source_user is {:.4f}\n".format(np.mean(distance_final_user_with_source_user))) 
            file.write("    The score between random_user is {:.4f}\n".format(np.mean(distance_random_user_rep))) 
            file.write("    The mean order of pos item in candidate is {:.4f}, in range is {:.4f}, and in sample is {:.4f}\n".format(sum(pos_order_in_candidate) / num_batch, sum(pos_order_in_range) / num_batch, sum(pos_order_in_sample) / num_batch))  
        model.eval()
#         if (args.dataset=='amazon_toy' and epoch >= 40) or (args.dataset=='amazon_game' and epoch >= 0):
        if epoch >= 20: # for fast training
            T = time.time() - t0
            t_test = evaluate_SASRec_V2_iteration_savedict(model, dataset, args, user_list, file_result_path, epoch)
            t3 = time.time()

            ndcg_list.append(t_test[2])
            hr_list.append(t_test[7])
            auc_list.append(t_test[10])
            loss_test_list.append(t_test[11])
    #         ipdb.set_trace()
            if curriculum_DNS_num == 0:
                print('epoch:%d, epoch_time: %.4f(s), total_time: %.4f(s), test:    NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, NDCG@50: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, HR@20: %.4f, HR@50: %.4f, AUC: %.4f, loss: %.4f, pos_score is %.4f, neg_RDNS_score is %.4f, neg_RNS_score is %.4f, Number_of_user in train: %d\n' % (epoch, t3-t1, t3-t0, t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5], t_test[6], t_test[7], t_test[8], t_test[9], t_test[10], t_test[11], sum(model_score_pos_sum) / sum(with_train_score_num), sum(model_score_neg_DNS_sum) / sum(with_train_score_num), sum(model_score_neg_RNS_sum) / sum(with_train_score_num), len(torch.unique(u_all_tensor))))
                with io.open(result_path + 'test_performance.txt', 'a', encoding='utf-8') as file:
                    file.write('epoch:%d, epoch_time: %.4f(s), total_time: %.4f(s), test:\n' % (epoch, t3-t1, t3-t0))
                    file.write('    NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, NDCG@50: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, HR@20: %.4f, HR@50: %.4f, AUC: %.4f, loss: %.4f, pos_score is %.4f, neg_RDNS_score is %.4f, neg_RNS_score is %.4f, Number_of_user in train: %d\n' % (t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5], t_test[6], t_test[7], t_test[8], t_test[9], t_test[10], t_test[11], sum(model_score_pos_sum) / sum(with_train_score_num), sum(model_score_neg_DNS_sum) / sum(with_train_score_num), sum(model_score_neg_RNS_sum) / sum(with_train_score_num), len(torch.unique(u_all_tensor))))
            else:
                print('epoch:%d, epoch_time: %.4f(s), total_time: %.4f(s), test:    NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, NDCG@50: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, HR@20: %.4f, HR@50: %.4f, AUC: %.4f, loss: %.4f, pos_score is %.4f, neg_RDNS_score is %.4f, neg_RNS_score is %.4f, Number_of_user in train: %d, DNS_prob: %.4f, source_DNS_prob: %.4f\n' % (epoch, t3-t1, t3-t0, t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5], t_test[6], t_test[7], t_test[8], t_test[9], t_test[10], t_test[11], sum(model_score_pos_sum) / sum(with_train_score_num), sum(model_score_neg_DNS_sum) / sum(with_train_score_num), sum(model_score_neg_RNS_sum) / sum(with_train_score_num), len(torch.unique(u_all_tensor)), sum(raw_DNS_num)/(sum(with_train_score_num)*curriculum_DNS_num),  sum(raw_DNS_source_num)/(sum(with_train_score_num)*max(curriculum_source_DNS_num,1))))
                with io.open(result_path + 'test_performance.txt', 'a', encoding='utf-8') as file:
                    file.write('epoch:%d, epoch_time: %.4f(s), total_time: %.4f(s), test:\n' % (epoch, t3-t1, t3-t0))
                    file.write('    NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, NDCG@50: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, HR@20: %.4f, HR@50: %.4f, AUC: %.4f, loss: %.4f, pos_score is %.4f, neg_RDNS_score is %.4f, neg_RNS_score is %.4f, Number_of_user in train: %d, DNS_prob: %.4f, source_DNS_prob: %.4f, outlier_prob:%.4f\n' % (t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5], t_test[6], t_test[7], t_test[8], t_test[9], t_test[10], t_test[11], sum(model_score_pos_sum) / sum(with_train_score_num), sum(model_score_neg_DNS_sum) / sum(with_train_score_num), sum(model_score_neg_RNS_sum) / sum(with_train_score_num), len(torch.unique(u_all_tensor)), sum(raw_DNS_num)/(sum(with_train_score_num)*curriculum_DNS_num),  sum(raw_DNS_source_num)/(sum(with_train_score_num)*max(curriculum_source_DNS_num,1)), sum(raw_outlier_source_num)/max(sum(raw_outlier_positive_num),1)))

            early_stopping(epoch, model, result_path, t_test)

            if early_stopping.counter > early_stopping.patience:
                print("Save in path:{}\n".format(result_path))
                print("Early stopping in the epoch {}, the NDCG@10: {:.4f}, HR@10: {:.4f}, AUC: {:.4f}, loss: {:.4f}\n".format(early_stopping.save_epoch, early_stopping.best_performance[2], early_stopping.best_performance[7], early_stopping.best_performance[10], early_stopping.best_performance[11]))
                print('NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, NDCG@50: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, HR@20: %.4f, HR@50: %.4f, AUC: %.4f, loss: %.4f\n' % (early_stopping.best_performance[0], early_stopping.best_performance[1], early_stopping.best_performance[2], early_stopping.best_performance[3], early_stopping.best_performance[4], early_stopping.best_performance[5], early_stopping.best_performance[6], early_stopping.best_performance[7], early_stopping.best_performance[8], early_stopping.best_performance[9], early_stopping.best_performance[10], early_stopping.best_performance[11]))
                with io.open(result_path + 'save_model.txt', 'a', encoding='utf-8') as file:
                    file.write("Early stopping in the epoch {}, the NDCG@10: {:.4f}, HR@10: {:.4f}, AUC: {:.4f}, loss_rec: {:.4f}\n".format(epoch, early_stopping.best_performance[2], early_stopping.best_performance[7], early_stopping.best_performance[10], early_stopping.best_performance[11]))
                    file.write('NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, NDCG@50: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, HR@20: %.4f, HR@50: %.4f, AUC: %.4f, loss: %.4f\n' % (early_stopping.best_performance[0], early_stopping.best_performance[1], early_stopping.best_performance[2], early_stopping.best_performance[3], early_stopping.best_performance[4], early_stopping.best_performance[5], early_stopping.best_performance[6], early_stopping.best_performance[7], early_stopping.best_performance[8], early_stopping.best_performance[9], early_stopping.best_performance[10], early_stopping.best_performance[11]))
                break

    
    sampler.close()
