# 获取到archor之后，比较archor之间的版本

import json
import os
import pickle
import csv
import torch
from tqdm import tqdm
import time
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-0.3*x))


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
#         ('1.0.1.4', '1.0.2.4')       :['1.0.2'] + [f'1.0.2.{i}' for i in range(5)],
#         ('1.0.1.4', None)       :['1.0.2'] + [f'1.0.2.{i}' for i in range(5)],
#         ('1.0.2.4', '3.0.16')        :[f"3.0.{i}" for i in range(0, 17) if i not in [3,13]],
#         ('3.0.16', '3.1.8')     :[f"3.1.{i}" for i in range(0, 9)],
#         ('3.1.8', '3.3.3')      :[f"3.2.{i}" for i in range(0, 5)] + [f"3.3.{i}" for i in range(0, 4)],
#         ('3.3.3', None)         :['3.4.0', '3.4.1', '3.5.0']
#     },
#     "libfreetype.so" : {
#         (None, '2.4.11')       : [f'2.4.{i}' for i in range(0, 9)] + [f'2.4.{i}' for i in range(10, 12)],
#         ('2.4.11', '2.5.5')    : ['2.4.12'] + [f'2.5.{i}' for i in range(1, 6)],
#         ('2.5.5', '2.12.1')    : ['2.7.1', '2.8', '2.8.1', '2.9.1'] + [f'2.10.{i}' for i in range(0, 5)] + ['2.11.1', '2.11.0', '2.12.0', '2.12.1'],
#         ('2.12.1', None)       : [f'2.13.{i}' for i in range(0, 4)] 
#     }
# }



#####################################################################
def read_json(json_path:str):
    with open(json_path, 'r')as f:
        data = json.load(f)
    return data

def read_pkl(file_path: str) -> list:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)


config = read_json('config.json')['dataset']
        
######################################################################


#######################################################################

def sort_dict_by_version(dictionary):
    """
    根据字典的键（版本号 a.b.c）对字典进行排序。

    :param dictionary: 字典，键为版本号，例如 {"1.2.3": "value1", "2.3.4": "value2"}
    :return: 排序后的字典
    """
    # 将版本号拆分为元组，便于排序
    def version_to_tuple(version):
        return tuple(map(int, version.split('.')))

    # 根据版本号对字典进行排序
    sorted_items = sorted(dictionary.items(), key=lambda x: version_to_tuple(x[0]))
    
    # 将排序后的结果转换为字典
    return dict(sorted_items)



def read_strings(file_path):
    return read_pkl(file_path)

def read_embs(file_path):
    data = read_pkl(file_path)
    if 'embedding ' in data:
        embs =  data['embedding ']
    elif 'embedding' in data:
        embs =  data['embedding']
    elif 'embeddings' in data:
        embs =  data['embeddings']
    else:
        raise KeyError(f"文件 {file_path} 中没有 'embedding ' 键")
    names = data['name']
    if 'imm_store' in data:
        imm_store = data['imm_store'] 
    else:
        imm_store = None
        
    return embs, names, imm_store

def get_archor_version():
    archor_version_path = os.path.join(ArchorDir, f"{Proj}", Work[0], Work[1], 'archor_ver.json')
    return read_json(archor_version_path)

def save_to_csv(data, filename):
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    except Exception as e:
        print(f"保存文件时出错: {e}")

def get_candidate_version(archor_version):
    if type(archor_version)==list:
        archor_version = tuple(archor_version)
    return VersionStore[archor_version]

def match_strings_single(tar_strings, candidate_strings):
    if len(candidate_strings) == 0:
        return None
    same_strings = set(tar_strings)&set(candidate_strings)
    match_ratio = len(same_strings)/len(candidate_strings)
    return match_ratio

def match_embs_single(tar_embs, candidate_embs, candidate_names):
    # 归一化向量
    tar_embs_normalized = torch.nn.functional.normalize(tar_embs, p=2, dim=1)
    candidate_embs_normalized = torch.nn.functional.normalize(candidate_embs, p=2, dim=1)
    
    similarities = torch.matmul(candidate_embs_normalized,  tar_embs_normalized.T) 
    similarities =  torch.nan_to_num(similarities, nan=0) 
    max_similarities = torch.max(similarities,  dim=1).values 
    max_indices = torch.argmax(similarities, dim=1)
    
    matched_indices = []
    # for candidate_feat_name in candidate_features.keys():
    #     for i, candidate_name in enumerate(candidate_names):
    #         if candidate_feat_name == candidate_name:
    #             max_index = max_indices[i]
    #             matched_indices.append(max_index.item())
    #             break
    return torch.mean(max_similarities).item(), matched_indices

import numpy as np
def match_features_single(tar_features, candidate_features, matched_names, idb_path):
    scores = []
    if not matched_names:
        return False
    candidate_names = list(candidate_features.keys())
    for i, matched_name in enumerate(matched_names):
        score = 0
        tar_feature = tar_features[(idb_path, matched_name)]
        candidate_feature = candidate_features[candidate_names[i]]
        if len(candidate_feature['call_args'][0]) == 0 and len(candidate_feature['imm_store'][0]) == 0 \
            and len(candidate_feature['imm_store'][1]) == 0 :
            continue

        matched_score_1 = 1 - abs(sum(candidate_feature['imm_store'][0] | candidate_feature['imm_store'][1]) - sum(tar_feature['imm_store'])) /\
                            (sum(candidate_feature['imm_store'][0] | candidate_feature['imm_store'][1]) + sum(tar_feature['imm_store']))/2\
                                if sum(candidate_feature['imm_store'][0] | candidate_feature['imm_store'][1]) > 0 else None
                            
        matched_score_2 = 1 - abs(sum(candidate_feature['call_args'][0]) - sum(tar_feature['call_args'])) / \
                            (sum(candidate_feature['call_args'][0]) + sum(tar_feature['call_args']))/2 if sum(candidate_feature['call_args'][0]) > 0 else None
                            
        tar_nodes_feat = np.mean(tar_feature['nodes_feat'], axis = 0)
        
        matched_score_3 = 1 - np.nanmean(2*np.abs(tar_nodes_feat - candidate_feature['nodes_feat'][0])/(tar_nodes_feat + candidate_feature['nodes_feat'][0]))
        
        matched_score_2 = None
        
        if np.isnan(matched_score_3):
            matched_score_3 = None
        

        for imm in candidate_feature['imm_store'][0]:
            if imm in tar_feature['imm_store']:
                score += 2
        for imm in candidate_feature['imm_store'][1]:
            if imm in tar_feature['imm_store']:
                score += 1
        # for imm in candidate_feature['imm_store'][1]:
        #     if imm in tar_feature['imm_store']:
        #         score -= 2 
        for call_arg in candidate_feature['call_args'][0]:
            if call_arg in tar_feature['call_args']:
                score += 1
        # for call_arg in candidate_feature['call_args'][1]:
        #     if call_arg in tar_feature['call_args']:
        #         score -= 1


        score = score / (len(candidate_feature['call_args'][0])+ 2*len(candidate_feature['imm_store'][0]) + len(candidate_feature['imm_store'][1])) \
            if (len(candidate_feature['call_args'][0]) + len(candidate_feature['imm_store'][0]) + len(candidate_feature['imm_store'][1])) > 0 else score
        # 对score, matched_score_1, matched_score_2, matched_score_3求平均，跳过None
        score_items = [score]
        for ms in [matched_score_1, matched_score_2, matched_score_3]:
            if ms is not None:
                score_items.append(ms)
        if score_items:
            score = sum(score_items) / len(score_items)
        
        
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0


def match_detail_all(tar_ver, candidate_versions):
    # 匹配主函数
    candidate_versions.sort()
    StringDir = os.path.join(DataDir, 'DBs', 'Strings', Proj)
    Detail_FeatureDir = os.path.join(DataDir, 'DBs', 'Detail_Features', Proj)
    EmbeddingDir = os.path.join(DataDir, 'DBs', 'Embedding', Proj)
    FeatDir = os.path.join(DataDir, 'DBs', 'Cfg', Proj)
    result = {}
    start_time = time.time()
    
    # 1 获取tar路径
    if '_' in Work[0]:
        lib_arch, tar_arch = Work[0].split('_')
        lib_opt = Work[1]
        tar_opt = Work[1]
    else:
        lib_opt, tar_opt =  Work[1].split('_')
        lib_arch = Work[0]
        tar_arch = Work[0]
    tar_string_path = os.path.join(StringDir, tar_arch, tar_opt, tar_ver, f"{Proj}.pkl")
    tar_strings = read_strings(tar_string_path)
    target_embs_path = os.path.join(EmbeddingDir, tar_arch, tar_opt, tar_ver, Proj, 'all_func.pkl')
    target_embs, tar_names, tar_imms = read_embs(target_embs_path)
    tar_feature_path = os.path.join(FeatDir, tar_arch, tar_opt, tar_ver, Proj, 'all_func.pkl')
    tar_features = read_pkl(tar_feature_path)
    idb_path = os.path.join('IDBs', Proj, tar_arch, tar_opt, tar_ver, f"{Proj}.i64")
    
    # 2 遍历candidate_versions
    max_ratio = 0
    match_ver = None
    for candidate_version in candidate_versions:
        # print(f"正在匹配 {tar_ver} 与 {candidate_version} ...")
        
        candidate_string_path = os.path.join(Detail_FeatureDir, candidate_version, 'strings.pkl')
        candidate_strings = read_strings(candidate_string_path)
        string_ratio = match_strings_single(tar_strings, candidate_strings)
        
        candidate_embs_path = os.path.join(Detail_FeatureDir, candidate_version, 'embedding.pkl')
        candidate_embs, candidate_names, candidate_imms = read_embs(candidate_embs_path)

        # candidate_feature_path = os.path.join(Detail_FeatureDir, candidate_version, 'features.pkl')
        # candidate_features = read_pkl(candidate_feature_path)
        
        if candidate_embs is None:
            emb_ratio = None
            indices = None
        else:
            # 确定list(candidate_features.keys())在tar_names中的索引
            emb_ratio, indices = match_embs_single(target_embs, candidate_embs, candidate_names)
            matched_names = [tar_names[i] for i in indices]
            # feature_ratio = match_features_single(tar_features, matched_names, idb_path)
            try:
                pass
                # print(f"\n正在匹配 {tar_ver} 与 {candidate_version} ...\n 字符串匹配率: {string_ratio:.4f},\n 嵌入匹配率: {emb_ratio:.4f},\n 特征匹配率: {feature_ratio:.4f}")
            except Exception as e:
                pass

            
            # if feature_ratio is False:
            #     _emb_ratio = emb_ratio
            #     emb_ratio = 0.9*emb_ratio
            # else:
            #     _emb_ratio = emb_ratio
            #     emb_ratio = (emb_ratio + feature_ratio) / 2 if emb_ratio is not None else feature_ratio
            
            alpha = 0.05*math.log(candidate_embs.shape[0])
            if len(candidate_strings) < 3:
                alpha = 0.15
            
        if string_ratio is None:
            string_ratio = 0.8

        if emb_ratio is None:
            match_ratio = 0.8*string_ratio
        else:
            
            match_ratio = (1-alpha)*string_ratio + alpha*emb_ratio
        if match_ratio > max_ratio:
            max_ratio = match_ratio
            match_ver = candidate_version
        # print(f"\n匹配 {tar_ver} 与 {candidate_version} 的结果:\n 字符串匹配率: {string_ratio:.4f},\n 嵌入匹配率: {_emb_ratio:.4f},\n 权重：{alpha:.4f},\n 特征匹配率: {feature_ratio:.4f},\n 综合匹配率: {(1-alpha)*string_ratio + alpha*_emb_ratio:.4f}")


    total_time = time.time() - start_time
    result[match_ver] = [max_ratio, lib_arch, lib_opt, tar_arch, tar_opt, total_time]
    return result

    

def save_result(result):
    # 将结果保存到csv中
    result = sort_dict_by_version(result)
    head = ['tar_ver', 'tar_arch', 'tar_opt', 'match_ver', 'match_arch', 'match_opt', 'result', 'time']
    csv_data = [head]
    for tar_ver, _match_info in result.items():
        for match_ver, match_info in _match_info.items():
            csv_line = [tar_ver, match_info[3], match_info[4], match_ver, match_info[1], match_info[2], False, match_info[5]]
            if tar_ver == match_ver:
                csv_line[-2] = True
            csv_data.append(csv_line)
    result_path = os.path.join(ResultDir, config['name'], f"{Proj}", Work[0], Work[1], 'rasult.csv')
    dirname = os.path.dirname(result_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    save_to_csv(csv_data, result_path)

def main():
    result = {}
    s_time = time.time()
    archor_versions_store = get_archor_version()
    for tar_ver, archor_versions in archor_versions_store.items():
        candidate_versions = []
        if not archor_versions:
            # 无archor_versions，全版本匹配
            archor_versions = [(None, None)]
        # archor_versions = [(None, None)]
        for archor_version in archor_versions:
            candidate_versions.extend(get_candidate_version(archor_version))
        
        result[tar_ver] = match_detail_all(tar_ver, candidate_versions)
    print(f"{Proj}  {_opts} {(time.time() - s_time)/len(archor_versions_store)}")
    for tar_ver, _match_info in result.items():
        for match_ver, match_info in _match_info.items():
            print(f"tar_ver: {tar_ver}\n tar_arch: {match_info[3]}\n tar_opt: {match_info[4]}\n\
                  \nmatch_ver: {match_ver}\n match_arch: {match_info[1]}\n match_opt: {match_info[2]}\n time: {match_info[5]:.2f}s\
                      \n" + "#"*20 + '\n')
    save_result(result)


####################################################################
# 配置区域
if __name__ == "__main__":
    ArchorDir = f'/sdb/Version_recg/results/Archor_Version/{config["name"]}'
    DataDir = f'/sdb/Version_recg/data/{config["name"]}'
    ResultDir = '/sdb/Version_recg/results/Detail_match'
    work_type = 'XO' # XA 跨架构 XO 跨优化
    _Proj = input("请输入要比较的项目名称（如libaws-c-common.so）：")
    
    for Proj in os.listdir(f'data/{config["name"]}/DBs/Embedding/'):
        if not _Proj:
            pass
        elif Proj != _Proj:
            continue
        ##################### DEBUG ########################
        # if Proj == 'cc1' or Proj == 'libcrypto.so.3':
        #     continue
        ####################################################
    
        archor_map_path = os.path.join(config['root'], config['name'], 'DBs', 'Archor', Proj, 'archor_map.pkl')
        VersionStore = read_pkl(archor_map_path)
        all_versions = []
        for archor_version ,  versions in VersionStore.items():
            all_versions.extend(versions)
        VersionStore[(None, None)] = all_versions
        
        string_ratio_path = os.path.join(DataDir, 'DBs', 'Detail_Features', Proj, 'string_ratio.pkl')
        # _string_ratio = read_pkl(string_ratio_path)
        
        if work_type == 'XO':

            # opts_store = [f'O{i}' for i in range(4)]
            opts_store = ['O0', 'O1', 'O2', 'O3']
            if Proj == 'libc.so.6':
                opts_store = ['O1', 'O2', 'O3']

            opt_pair = []
            left_opt = 'O2'
            for right_opt in opts_store:
                opt_pair.append([left_opt, right_opt])
            # pbar = tqdm(total=len(opt_pair))
            for _opts in opt_pair:
                Work = ['X64', f"{_opts[0]}_{_opts[1]}"]
                try:
                    s_time = time.time()
                    main()
                    
                except Exception as e:
                    print(f"Error processing {Work}: {e}")
                # pbar.update(1)
            # pbar.close()
        elif work_type == 'XA':

            arch_pair = []
            left_arch = 'X64'
            for right_arch in ['X86', 'ARM']:
                arch_pair.append([left_arch, right_arch])
            pbar = tqdm(total=len(arch_pair))
            for _archs in arch_pair:
                Work = [f"{_archs[0]}_{_archs[1]}", 'O2']
                try:
                    main()
                except Exception as e:
                    print(f"Error processing {Work}: {e}")
                pbar.update(1)
            pbar.close()

