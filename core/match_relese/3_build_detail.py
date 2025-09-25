# 比较detail version 与 archor version 之间的细粒度差异
import os
import pickle
import json
import torch
import itertools
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from scipy.spatial.distance import cdist



#####################################################################
# 全局变量
# Proj = 'libaws-c-common.so' # libcrypto.so.3 libfreetype.so libcrypto.so
# VersionStore = {
#     'cc1'   : {
#         (None, '7.5.0')     : [f"7.{i}.0" for i in range(2,6)],
#         ('7.5.0', '8.5.0')  : [f"8.{i}.0" for i in range(1,6)],
#         ('8.5.0', '9.5.0')  : [f"9.{i}.0" for i in range(1,6)],
#         ('9.5.0', '10.5.0') : [f"10.{i}.0" for i in range(1,6)],
#         ('10.5.0', '11.5.0'): [f"11.{i}.0" for i in range(1,6)],
#         ('11.5.0', '12.4.0'): [f"12.{i}.0" for i in range(1,5)],
#         ('12.4.0', '13.3.0'): [f"13.{i}.0" for i in range(1,4)],
#         ('13.3.0', None)    : [f"14.{i}.0" for i in range(1,3)]    
#         },
#     'libcrypto.so.3' : {
#         (None, '1.0.0.3')       :['1.0.0', '1.0.0.0', '1.0.0.1', '1.0.0.2', '1.0.0.3'],
#         ('1.0.0.3', '1.0.1.4')  :['1.0.1.1', '1.0.1.2', '1.0.1.3', '1.0.1.4'],
#         ('1.0.1.4', None)       :['1.0.2'] + [f'1.0.2.{i}' for i in range(5)],
#         (None, '3.0.16')        :[f"3.0.{i}" for i in range(0, 17) if i not in [3,13]],
#         ('3.0.16', '3.1.8')     :[f"3.1.{i}" for i in range(0, 9)],
#         ('3.1.8', '3.3.3')      :[f"3.2.{i}" for i in range(0, 5)] + [f"3.3.{i}" for i in range(0, 4)],
#         ('3.3.3', None)         :['3.4.0', '3.4.1', '3.5.0']
#     },
#     "libfreetype.so" : {
#         (None, '2.4.11')       : [f'2.4.{i}' for i in range(0, 9)] + [f'2.4.{i}' for i in range(10, 12)],
#         ('2.4.11', '2.5.5')    : ['2.4.12'] + [f'2.5.{i}' for i in range(1, 6)],
#         ('2.5.5', '2.12.1')    : ['2.7.1', '2.8', '2.8.1', '2.9.1'] + [f'2.10.{i}' for i in range(0, 5)] + ['2.11.1', '2.11.0','2.12.0', '2.12.1'],
#         ('2.12.1', None)       : [f'2.13.{i}' for i in range(0, 4)] 
#     }
# }

#######################################################################
def read_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_pickle_file(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
        
def get_all_files_in_dir(dir_path):
    all_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

        
######################################################################


#######################################################################

def compare_strings(version, archor_version):
    tar_string_path = os.path.join(String_Root, version, f'{Proj}.pkl')
    archor_string_path = os.path.join(String_Root, archor_version, f'{Proj}.pkl')
    tar_data = read_pickle_file(tar_string_path)
    archor_data = read_pickle_file(archor_string_path)
    unique_strings = set(tar_data) - set(archor_data)
    # save_pickle_file(os.path.join(Save_Root, version, 'strings.pkl'), unique_strings)
    # 查看其他优化是否存在
    for opt in ['O0', 'O1', 'O3']:
        opt_tar_string_path = os.path.join(String_Root.replace('O2', opt), version, f'{Proj}.pkl')
        if os.path.exists(opt_tar_string_path) :
            opt_tar_data = read_pickle_file(opt_tar_string_path)
            unique_strings &= set(opt_tar_data)  
    
    
    return unique_strings

import numpy as np
def compare_node_feat(cfgs):
    # 比较Cfg中的nodes_feat
    
    ret = [[], []]
    for i in range(1, len(cfgs)-1):
        cfg1 = cfgs[0]
        cfg2 = cfgs[i]
        diff = np.abs(np.mean(cfg1['nodes_feat'], axis=0) - np.mean(cfg2['nodes_feat'], axis=0))
        ret[0].append(diff)
        
    for i in range(1, len(cfgs)-1):
        cfg1 = cfgs[-1]
        cfg2 = cfgs[i]
        diff = np.abs(np.mean(cfg1['nodes_feat'], axis=0) - np.mean(cfg2['nodes_feat'], axis=0))
        ret[1].append(diff)
    
    compare_index_level1 = np.where((ret[0][0] - ret[0][1]) < 0)[0]
    compare_index_level2 = np.where((ret[1][0] - ret[1][1]) > 0)[0]
        
    return np.intersect1d(compare_index_level1, compare_index_level2).tolist()


def compare_imm_call_feat(cfgs):
    # 比较Cfg中的imm_store和call_args特征
    tar_imm = set(cfgs[0]['imm_store']) & set(cfgs[1]['imm_store'])
    tar_call = set(cfgs[0]['call_args']) & set(cfgs[1]['call_args'])
    archor_imm = set(cfgs[2]['imm_store']) | set(cfgs[3]['imm_store'])
    archor_call = set(cfgs[2]['call_args']) | set(cfgs[3]['call_args'])
    ret_imm = tar_imm - archor_imm
    ret_call = tar_call - archor_call
    return ret_imm, ret_call

def keep_indexed_elements(arr, index_list):
    """
    对输入的 numpy 数组进行处理，保留索引在 index_list 中的元素，其他元素设置为 nan。

    参数:
    arr (np.array): 输入的 numpy 数组。
    index_list (list): 需要保留的索引列表。

    返回:
    np.array: 处理后的数组，索引在 index_list 中的元素保留，其他元素为 nan。
    """
    # 创建一个与原始数组形状相同的 nan 数组
    result = np.full_like(arr, np.nan, dtype=float)
    
    # 将索引列表中的元素复制到结果数组中
    result[index_list] = arr[index_list]
    
    return result

def compare_features(version, archor_version, names):
    # 不能只统计修改的部分，其他部分也要统计
    
    feat_res = {}
    tar_feature_path = os.path.join(Feature_Root, version, Proj, 'all_func.pkl')
    archor_feature_path = os.path.join(Feature_Root, archor_version, Proj, 'all_func.pkl')
    control_tar_feature_path = os.path.join(Feature_Root.replace('O2', 'O0'), version, Proj, 'all_func.pkl')
    control_archor_feature_path = os.path.join(Feature_Root.replace('O2', 'O0'), archor_version, Proj, 'all_func.pkl')
    
    tar_data = read_pickle_file(tar_feature_path)
    archor_data = read_pickle_file(archor_feature_path)
    control_tar_data = read_pickle_file(control_tar_feature_path)
    control_archor_data = read_pickle_file(control_archor_feature_path)
    

    
    for name in names:
        tar_key = (f'IDBs/{Proj}/X64/O2/{version}/{Proj}.i64', name)
        control_tar_key = (f'IDBs/{Proj}/X64/O0/{version}/{Proj}.i64', name)
        archor_key = (f'IDBs/{Proj}/X64/O2/{archor_version}/{Proj}.i64', name)
        control_archor_key = (f'IDBs/{Proj}/X64/O0/{archor_version}/{Proj}.i64', name)
        
        if tar_key not in tar_data.keys() or archor_key not in archor_data.keys() or \
           control_tar_key not in control_tar_data.keys() or control_archor_key not in control_archor_data.keys():
            continue
        tar_feat = tar_data[tar_key]
        archor_feat = archor_data[archor_key]
        control_tar_feat = control_tar_data[control_tar_key]
        control_archor_feat = control_archor_data[control_archor_key]
        
        cfgs = [tar_feat, control_tar_feat, control_archor_feat, archor_feat]
        node_feat_index = compare_node_feat(cfgs)
        
        
        # 1. imm_store
        imm_feat_tar, call_arg_feat_tar = compare_imm_call_feat(cfgs)
        if len(imm_feat_tar) + len(call_arg_feat_tar) + len(node_feat_index) == 0:
            continue
        # 筛选出版本差异 > 编译差异的节点特征
        tar_nodes_feat = np.mean(tar_feat['nodes_feat'], axis=0)
        tar_nodes_feat = keep_indexed_elements(tar_nodes_feat, node_feat_index)

        archor_nodes_feat = np.mean(archor_feat['nodes_feat'], axis=0)
        archor_nodes_feat = keep_indexed_elements(archor_nodes_feat, node_feat_index)

        
        feat_res[name] = {
            'imm_store' : [imm_feat_tar, set(tar_feat['imm_store']) - imm_feat_tar],
            'call_args' : [call_arg_feat_tar],
            'nodes_feat' : [tar_nodes_feat, archor_nodes_feat],
        }
    return feat_res
    

def compare_embs(version, archor_version):
    tar_emb_path = os.path.join(Emb_Root, version, Proj, 'all_func.pkl')
    archor_emb_path = os.path.join(Emb_Root, archor_version, Proj, 'all_func.pkl')
    tar_data = read_pickle_file(tar_emb_path)
    archor_data = read_pickle_file(archor_emb_path)
    
    # 1 寻找独特函数
    tar_names = [name for name in tar_data['name'] if not (name.startswith("sub_") or name.startswith("output_") or
                                                       name.startswith("gen_split_") or name.startswith("gen_peephole2_") or name.startswith("pattern"))]
    archor_names = [name for name in archor_data['name'] if not (name.startswith("sub_") or name.startswith("output_") or
                                                       name.startswith("gen_split_") or name.startswith("gen_peephole2_") or name.startswith("pattern"))]
    same_names = set(tar_names) & set(archor_names)
    Unique_names = list(set(tar_names) - same_names)
    
    # 2 寻找低相似度函数
    tar_emb = torch.stack([tar_data['embedding '][tar_data['name'].index(name)] for name in same_names])
    archor_emb = torch.stack([archor_data['embedding '][archor_data['name'].index(name)] for name in same_names])
    # 归一化向量
    tar_embeddings_normalized = torch.nn.functional.normalize(tar_emb, p=2, dim=1)
    archor_embeddings_normalized = torch.nn.functional.normalize(archor_emb, p=2, dim=1)
    # 计算余弦相似度
    similarities = torch.sum(tar_embeddings_normalized * archor_embeddings_normalized, dim=1).tolist()
    Unique_names.extend([name for name, similarity in zip(same_names, similarities) if similarity < 0.95])
    if len(Unique_names) < 3:
        Unique_names.extend([name for name, similarity in zip(same_names, similarities) if similarity < 0.99])
        Unique_names = list(set(Unique_names))
    Unique_embs = [tar_data['embedding '][tar_data['name'].index(name)] for name in Unique_names]
    if len(Unique_embs) == 0:
        Unique_embs = None
    else:
        Unique_embs = torch.stack(Unique_embs)
    result = {
        'name'  : Unique_names,
        'embeddings'    : Unique_embs,
    }
    # save_path = os.path.join(Save_Root, version, 'embedding.pkl')
    # save_pickle_file(save_path, result)
    return result
    
def compare_versions(version, archor_version, _archor_version):
    
    
    embs = compare_embs(version, archor_version)
    # feats = compare_features(version, archor_version, embs['name'])
    
    _embs = compare_embs(version, _archor_version)
    # _feats = compare_features(version, _archor_version, _embs['name'])
    # 取embs和_embs的并集
    all_names = set(embs['name']) | set(_embs['name'])
    all_embs = [embs['embeddings'][embs['name'].index(name)] if name in embs['name'] else _embs['embeddings'][_embs['name'].index(name)] for name in all_names]
    if len(all_embs) == 0:
        all_embs = None
    else:
        all_embs = torch.stack(all_embs)
    result_embs = {
        'name'  : list(all_names),
        'embeddings'    : all_embs,
    }
    # save_path = os.path.join(Save_Root, version, 'embedding.pkl')
    # save_pickle_file(save_path, result_embs)

    strings = compare_strings(version, archor_version)
    _strings = compare_strings(version, _archor_version)
    # 取strings和_strings的并集
    all_strings = set(strings) | set(_strings)
    # save_path = os.path.join(Save_Root, version, 'strings.pkl')
    # save_pickle_file(save_path, all_strings)
    
    # for name, info in feats.items():
    #     info['imm_store'][0] = info['imm_store'][0] | _feats[name]['imm_store'][0] if name in _feats else info['imm_store'][0]
    #     info['imm_store'][1] = info['imm_store'][1] | _feats[name]['imm_store'][1] if name in _feats else info['imm_store'][1]
    #     info['call_args'][0] = info['call_args'][0] | _feats[name]['call_args'][0] if name in _feats else info['call_args'][0]
    feats = None

    return result_embs, all_strings, feats

def save_results(version, result_embs, all_strings, feats):
    save_path = os.path.join(Save_Root, version)
    os.makedirs(save_path, exist_ok=True)
    
    emb_save_path = os.path.join(save_path, 'embedding.pkl')
    save_pickle_file(emb_save_path, result_embs)
    
    strings_save_path = os.path.join(save_path, 'strings.pkl')
    save_pickle_file(strings_save_path, all_strings)
    
    feats_save_path = os.path.join(save_path, 'features.pkl')
    save_pickle_file(feats_save_path, feats)
    
def caculate_ratio():
    # 计算字符串阈值
    Archor_map_path = f'data/{config["name"]}/DBs/Archor/{Proj}/archor_map.pkl'
    VersionStore = read_pickle_file(Archor_map_path)
    Version_list = list(VersionStore.values())
    
    all_string_paths = get_all_files_in_dir(String_Root)
    all_unique_string_paths = get_all_files_in_dir(Save_Root)
    all_unique_string_paths = [path for path in all_unique_string_paths if path.endswith('strings.pkl')]
    ratios = []
    for string_path in all_string_paths:
        tar_version = string_path.split('/')[-2]
        candidate_versions = []
        for versions in Version_list:
            if tar_version in versions:
                candidate_versions = versions
                break
        for unique_string_path in all_unique_string_paths:
            
            unique_version = unique_string_path.split('/')[-2]
            if unique_version not in candidate_versions:
                continue
            strings = read_pickle_file(string_path)
            unique_strings = read_pickle_file(unique_string_path)
            if len(unique_strings) == 0:
                continue
            ratio = len(unique_strings & set(strings)) / len(unique_strings)
            ratios.append(ratio)
    ratio = np.mean(ratios) + 0.5* np.std(ratios)
    print(f"字符串相似度阈值: {ratio}")
    save_path = os.path.join(Save_Root, 'string_ratio.pkl')
    save_pickle_file(save_path, ratio)
            

def main():
    result = {}
    pbar = tqdm(total=len(VersionStore), desc="Processing versions", ncols=100)
    for archor_pair, versions in VersionStore.items():
        archor_version = archor_pair[0]
        _archor_version = archor_pair[1] 
        if archor_pair[0] == None:
            archor_version = versions[0]
        if archor_pair[1] == None:
            _archor_version = versions[-1]
        
        for idx, version in enumerate(versions):
            if version == archor_version:
                new_versions = [_ver for _ver in versions if _ver != version]
                archor_version = new_versions[0]
            if version == _archor_version:
                new_versions = [_ver for _ver in versions if _ver != version]
                _archor_version = new_versions[-1]
            if version == archor_version or version == _archor_version:
                raise ValueError(f"版本 {version} 与锚版本 {archor_version} 或 {_archor_version} 相同")
            result_embs, all_strings, feats = compare_versions(version, archor_version, _archor_version)
            
            # 新增局部特征，要求比较version与相邻版本之间的细粒度差异
            if idx == 0:
                archor_version = versions[0]
                _archor_version = versions[1]
            elif idx == len(versions) - 1:
                archor_version = versions[-2]
                _archor_version = versions[-1]
            else:
                archor_version = versions[idx - 1]
                _archor_version = versions[idx + 1]
            _result_embs, _all_strings, _feats = compare_versions(version, archor_version, _archor_version)
            
            # 合并
            for i, name in enumerate(_result_embs['name']):
                if name not in result_embs['name']:
                    result_embs['name'].append(name)
                    result_embs['embeddings'] = torch.cat((result_embs['embeddings'], _result_embs['embeddings'][i].unsqueeze(0)), dim=0)
            all_strings = set(_all_strings) | set(all_strings)
            # for name, info in _feats.items():
            #     if name not in feats:
            #         feats[name] = info
            #     else:
            #         feats[name]['imm_store'][0] |= info['imm_store'][0]
            #         feats[name]['imm_store'][1] |= info['imm_store'][1]
            #         feats[name]['call_args'][0] |= info['call_args'][0]
                    # feats[name]['call_args'][1] |= info['call_args'][1]
            

            save_results(version, result_embs, all_strings, feats)
        pbar.update(1)
    pbar.close()

if __name__ == "__main__":
    
    #################################################
    # 全局变量
    config = read_json_file('config.json')['dataset']
    _Proj = input("请输入要比较的项目名称（如libaws-c-common.so）：")
    proj_list = os.listdir(f'data/{config["name"]}/DBs/Embedding/')
    
    for Proj in proj_list:
        # if Proj == 'cc1':
        #     continue
        
        if not _Proj:
            pass
        elif Proj != _Proj:
            continue
    
        archor_map_path = os.path.join(config['root'], config['name'], 'DBs', 'Archor', Proj, 'archor_map.pkl')
        VersionStore = read_pickle_file(archor_map_path)
        
        #################### 配置区域 #################### 
        # Proj in line 6
        Feature_Root = f"data/{config['name']}/DBs/Cfg/{Proj}/X64/O2"
        String_Root = f'data/{config["name"]}/DBs/Strings/{Proj}/X64/O2'
        Emb_Root = f'data/{config["name"]}/DBs/Embedding/{Proj}/X64/O2'
        Save_Root = f'data/{config["name"]}/DBs/Detail_Features/{Proj}/'
        #################################################

        main()
        caculate_ratio()