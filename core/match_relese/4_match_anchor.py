import os
import pickle
import json
import math
import torch
import warnings
import time
from packaging  import version as Version
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from distutils.version import LooseVersion

Threshold = 1.0
Threshold_sim = 0.6

def read_json(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data
config = read_json('config.json')['dataset']

def write_json(data, path):
    with open(path, 'w') as json_file:
        json.dump(data, json_file)

def get_folder_file_dict(target_dir):
    folder_file_dict = {}
    for root, dirs, files in os.walk(target_dir):
        # Only consider the first level subdirectories
        if root == target_dir:
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                folder_file_dict[dir_name] = os.listdir(dir_path)
            break
    return folder_file_dict

def read_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def match_one(target_data, lib_data):
    # 方案1 逐行匹配效率低
    result = {}
    target_embeddings = target_data['embedding ']
    lib_embeddings = lib_data['embedding ']
    lib_names = lib_data['name']

    for i, target_embedding in enumerate(target_embeddings):
        max_similarity = -1
        best_match_name = None
        for j, lib_embedding in enumerate(lib_embeddings):
            similarity = cosine_similarity(target_embedding.unsqueeze(0), lib_embedding.unsqueeze(0)).item()
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_name = lib_names[j]
        result[target_data['name'][i]] = {best_match_name: max_similarity}
    return result

def match_one_ii(target_data, lib_data):
    # 方案2 使用矩阵运算
    # 获取详细数据
    
    target_emb = target_data["embedding "] 
    lib_emb = lib_data["embedding "] 
    target_names = target_data["name"] 
    lib_names = lib_data["name"] 
    
    # 归一化向量 
    target_emb_normalized = torch.nn.functional.normalize(target_emb,  p=2, dim=1) 
    lib_emb_normalized = torch.nn.functional.normalize(lib_emb,  p=2, dim=1) 
    
    # 计算余弦相似度 
    cosine_similarities = torch.matmul(target_emb_normalized,  lib_emb_normalized.T) 
    
    # 找出相似度最高的行 
    max_indices = torch.argmax(cosine_similarities,  dim=1) 
    max_similarities = torch.max(cosine_similarities,  dim=1).values 
    
    # 输出结果 
    output = {} 
    for i in range(len(target_names)): 
        target_name = target_names[i] 
        max_index = max_indices[i].item() 
        lib_name = lib_names[max_index] 
        similarity = max_similarities[i].item() 
        output[target_name] = {lib_name: similarity} 
    return output

def match_one_rough(target_data, lib_data):
    # 获取粗略信息
    # 获取目标数据 
    target_emb = target_data["embedding "] 
    lib_emb = lib_data["embedding "] 
    target_names = target_data["name"] 
    lib_names = lib_data["name"] 
    
    # 归一化向量 
    target_emb_normalized = torch.nn.functional.normalize(target_emb,  p=2, dim=1) 
    lib_emb_normalized = torch.nn.functional.normalize(lib_emb,  p=2, dim=1) 
    
    # 计算余弦相似度 
    cosine_similarities = torch.matmul(target_emb_normalized,  lib_emb_normalized.T) 
    
    # 找出相似度最高的行 
    max_indices = torch.argmax(cosine_similarities,  dim=1) 
    max_values = torch.max(cosine_similarities,  dim=1).values 

    # 计算结果指标 
    avg_similarity = torch.mean(max_values).item() 
    count_above_threshold = (max_values > Threshold_sim).sum().item()
    
    return {
        'ratio': count_above_threshold / len(lib_names),
        "mean": avg_similarity,
        "num": count_above_threshold ,
        'total': len(target_names),
        'total_lib': len(lib_names),
        
    }
    
def match_sen_func(target_data, lib_embs, lib_names, lib_immstore):
    # 方案2 使用矩阵运算
    # 获取详细数据
    
    target_emb = target_data["embedding "] 
    target_names = target_data["name"] 
    target_immstore = target_data["imm_store"]
    tar_name_imm_map = {
        name: imm for name, imm in zip(target_names, target_immstore)
    }
    lib_name_imm_map = {_type : {} for _type in ['deleted', 'added']}
    for _type in ['deleted', 'added']:
        for name, imm in zip(lib_names[_type], lib_immstore[_type]):
            lib_name_imm_map[_type][name] = imm
    
    # 归一化向量 
    target_emb_normalized = torch.nn.functional.normalize(target_emb,  p=2, dim=1) 
    
    _types = ['deleted', 'added']
    result = {}
    for _type in _types:
        lib_emb = lib_embs[_type]
        lib_name = lib_names[_type]
        lib_emb_normalized = torch.nn.functional.normalize(lib_emb,  p=2, dim=1) 
        
        # 计算余弦相似度 
        cosine_similarities = torch.matmul(lib_emb_normalized,  target_emb_normalized.T) 
        
        # 找出相似度最高的行 
        max_indices = torch.argmax(cosine_similarities,  dim=1) 
        max_similarities = torch.max(cosine_similarities,  dim=1).values 
        
        # 找出相似度top-5的行
        top_k = 5
        top_k_indices = torch.topk(cosine_similarities, top_k, dim=1).indices
        top_k_similarities = torch.topk(cosine_similarities, top_k, dim=1).values
        
        # 获取top-k结果
        top_k_output = {}
        scores_output = {}
        allready_matched_func = []
        for i in range(len(lib_name)):
            _lib_name = lib_name[i]
            lib_imm = lib_name_imm_map[_type][_lib_name]
            top_k_matches = {}
            scores = {}
            for k in range(top_k):
                target_index = top_k_indices[i][k].item()
                target_name = target_names[target_index]

                tar_imm = tar_name_imm_map[target_name]
                common_elements_count = len(set(lib_imm).intersection(set(tar_imm)))
                score = common_elements_count / len(lib_imm)
                
                similarity = top_k_similarities[i][k].item()
                if math.isnan(similarity):
                    continue
                
                if similarity > score:
                    similarity = (score + similarity)/2
                    
                # if similarity < 0.5:
                #     continue
                
                #########################
                # 一个target_func只允许匹配一次
                # 后续再进行修改，现在先试用优先原则
                # if target_name in allready_matched_func:
                #     continue
                # else:
                #     allready_matched_func.append(target_name)
                #########################
                
                top_k_matches[target_name] = similarity
                scores[target_name] = score
            if not top_k_matches:
                continue
            top_k_output[_lib_name] = top_k_matches
            scores_output[_lib_name] = scores

        # 在top-k中筛选出得分最高的样本
        max_score_output = {}
        for lib_func, matched_info in top_k_output.items():
            if not matched_info:
                continue
            # 找到得分最高的函数
            max_score_func = max(matched_info, key=matched_info.get)
            max_score = matched_info[max_score_func]
            # 如果得分大于阈值，则加入结果
            max_score_output[lib_func] = {max_score_func: max_score}

        result[_type] = max_score_output
        # 输出结果 
        
        # DBG 计算误报率
        # count = {
        #     'deleted': 0, 'added': 0, 'changed': 0
        # }
        # for _type, _type_info in result.items():
            
        #     for lib_func, matched_info in _type_info.items():
        #         if lib_func == list(matched_info.keys())[0]:
        #             count[_type] += 1
        #     count[_type] = count[_type] / len(_type_info)
    return result
    
    
def calculate_destance(matched_ratio_all, matched_sim_all):
    for ver, ratios in matched_ratio_all.items():
        # deleted = 1-ratios['deleted']
        # added = 1-ratios['added']
        # changed = 1-ratios['changed']
        # distance = (deleted**2 + added**2 + changed**2)**0.5
        matched_ratio_all[ver]['dist'] = 0
        matched_ratio_all[ver]['score'] = 0.8*matched_sim_all[ver]['score'] + 0.2*matched_ratio_all[ver]['score']
        
def plot_da_ratio(matched_ratio_all, tar_ver):
    # 绘制D/A Ratio曲线
    import matplotlib.pyplot as plt

    # Extract keys and corresponding DA_ratio and dist values
    keys = sorted(matched_ratio_all.keys(), key=LooseVersion)
    # keys = sorted(result.keys())
    da_ratios = [matched_ratio_all[key]['score'] for key in keys]

    # Save the plot as a .png file
    plt.figure(figsize=(10, 5))
    plt.plot(keys, da_ratios, marker='o')
    plt.axhline(y=Threshold, color='red', linestyle='--')
    plt.xlabel('version')
    plt.ylabel('DA Ratio')
    plt.title(f'Target Version: {tar_ver}')
    plt.legend()
    # plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_dir = f'result_pic/test/{proj}/{_opts[0]}_vs_{_opts[1]}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'da_ratio_dist_plot_{tar_ver}.png'))
    # plt.savefig(f'result_pic/cc1_O1_vs_O2/da_ratio_dist_plot_{tar_ver}.png')
    
    return 

def verify_archor_ver(archor_vers, matched_ratio_all):
    # 验证archor_vers
    # 1. 对于archor_vers中的一个archor_ver:(ver1, ver2)，
    #     统计matched_ratio_all中小于ver1版本
    #     计算小于ver1版本的score小于1的比例
    #     统计matched_ratio_all中大于ver2版本
    #     计算大于ver2版本的score大于1的比例
    # 2. 如果比例大于0.5，则认为archor_ver有效
    # 3. 如果archor_ver无效，则删除archor_ver
    
    for archor_ver in archor_vers:
        ver1, ver2 = archor_ver
        if ver1 is None:
            # 统计matched_ratio_all中大于ver2版本
            count = 0
            for ver, ratios in matched_ratio_all.items():
                if LooseVersion(ver) >= LooseVersion(ver2):
                    if ratios['score'] > Threshold:
                        count += 1
            ratio = count / len(matched_ratio_all)
        elif ver2 is None:
            # 统计matched_ratio_all中小于ver1版本
            count = 0
            for ver, ratios in matched_ratio_all.items():
                if LooseVersion(ver) <= LooseVersion(ver1):
                    if ratios['score'] < Threshold:
                        count += 1
            ratio = count / len(matched_ratio_all)
        else:
            # 统计matched_ratio_all中小于ver1版本和大于ver2版本
            count = 0
            for ver, ratios in matched_ratio_all.items():
                if LooseVersion(ver) <= LooseVersion(ver1):
                    if ratios['score'] < Threshold:
                        count += 1
                if LooseVersion(ver) >= LooseVersion(ver2):
                    if ratios['score'] > Threshold:
                        count += 1
            ratio = count / len(matched_ratio_all)
        if ratio < 0.7:
            archor_vers.remove(archor_ver)
    return archor_vers

def get_archor_ver(matched_ratio_all):
    archor_vers = []
    # 将matched_ratio_all按照version排序
    sorted_versions = sorted(matched_ratio_all.keys(), key=LooseVersion)
    matched_ratio_all = {ver: matched_ratio_all[ver] for ver in sorted_versions}
    keys = list(matched_ratio_all.keys())
    for i in range(len(keys) - 1):
        current_key = keys[i]
        next_key = keys[i + 1]
        current_value = matched_ratio_all[current_key]
        next_value = matched_ratio_all[next_key]
        
        if current_value['score'] <= Threshold and next_value['score'] > Threshold:
            archor_vers.append((current_key, next_key))
    pass
    # 如果matched_ratio_all第一个和最后一个版本的score都大于1，则添加(None, 第一个版本)
    if  matched_ratio_all[keys[0]]['score'] >= Threshold and\
        matched_ratio_all[keys[-1]]['score'] >= Threshold:
        archor_vers.append((None, keys[0]))
        
    # 如果matched_ratio_all第一个和最后一个版本的score都小于1，则添加(最后一个版本, None)
    if  matched_ratio_all[keys[0]]['score'] < Threshold and\
        matched_ratio_all[keys[-1]]['score'] < Threshold:
        archor_vers.append((keys[-1], None))
    archor_vers = verify_archor_ver(archor_vers, matched_ratio_all)
    return archor_vers



def match_OSS(target_file, lib_dir):
    global Threshold
    target_data = read_pickle_file(target_file)
    lib_files = os.listdir(lib_dir)
    lib_files.sort()
    matched_result_all = {}
    matched_ratio_all = {}
    matched_sim_all = {}
    # pbar = tqdm(total=len(lib_files)//3, ncols=100)
    for lib_file in lib_files:
        # if lib_file.startswith('3'):
        #     continue
        if '_emb' in lib_file or 'imm' in lib_file:
            continue
        version = lib_file[:-4]
        matched_ratio_all[version] = {}
        matched_sim_all[version] = {}
        # pbar.set_description(version)
        # pbar.update(1)
        emb_file = os.path.join(lib_dir, f"{version}_emb.pkl")
        
        ######################
        # print(f"\ntar_ver : {tar_ver}, lib_ver : {version}\n")
        ######################
        
        if os.path.isfile(emb_file):
            lib_embs = read_pickle_file(emb_file)
            lib_names = read_pickle_file(os.path.join(lib_dir, lib_file))
            lib_immstore = read_pickle_file(os.path.join(lib_dir, f"{version}_immstore.pkl"))
            # Threshold = max(1+0.05*len(lib_names['deleted'])/len(lib_names['added']), Threshold)
            result = match_sen_func(target_data, lib_embs, lib_names, lib_immstore)

        for _type, matched_result in result.items():
            keys_to_delete = []
            sim_sum = 0
            for lib_func, matched_info in matched_result.items():
                sim_sum += list(matched_info.values())[0]
                keys_to_delete.append((lib_func, [matched_func for matched_func, similarity in matched_info.items() if similarity < 0.75]))
            for lib_func, funcs_to_delete in keys_to_delete:
                for matched_func in funcs_to_delete:
                    del matched_result[lib_func]

            matched_ratio_all[version][_type] = len(matched_result) / len(lib_names[_type])
            matched_sim_all[version][_type] = sim_sum / len(lib_names[_type])          
            
        if matched_ratio_all[version]['added'] == 0:
            matched_ratio_all[version]['added'] = 0.01
        if matched_sim_all[version]['added'] == 0:
            matched_sim_all[version]['added'] = 0.01
        matched_ratio_all[version]['score'] = matched_ratio_all[version]['deleted'] / matched_ratio_all[version]['added']
        matched_sim_all[version]['score'] = matched_sim_all[version]['deleted'] / matched_sim_all[version]['added']
        calculate_destance(matched_ratio_all, matched_sim_all)
        
        matched_result_all[version] = result
        
    # pbar.close()

    # plot_da_ratio(matched_ratio_all, tar_ver)
    
    arch_ver = get_archor_ver(matched_ratio_all)
    return arch_ver

def compare_versions(tar_ver, arch_ver):
    # 判斷tar_ver是否在arch_ver中間
    result = [False, False]
    if arch_ver[0] == None:
        result[0] = True
    elif Version.parse(arch_ver[0]) <= Version.parse(tar_ver):
        result[0] = True
    if arch_ver[1] == None:
        result[1] = True
    elif Version.parse(arch_ver[1]) >= Version.parse(tar_ver):
        result[1] = True
    return result[0] and result[1]
    
def caculate_result(arch_ver_dict):
    # 計算最終結果
    result = {}
    for tar_ver, arch_vers in arch_ver_dict.items():
        result[tar_ver] = 0
        for arch_ver in arch_vers:
            if compare_versions(tar_ver, arch_ver):
                result[tar_ver] = 1
        if len(arch_vers) == 0:
            result[tar_ver] = -1
        else:
            result[tar_ver] = result[tar_ver]/len(arch_vers)
    return result
            

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    # proj = 'libaws-c-common.so' # libcrypto.so.3  cc1 libfreetype.so libcrypto.so
    _proj = input("请输入项目名称：")
    work_type = 'XO' # 'XO'表示交叉编译，'XA'表示交叉架构 'XX'表示同时交叉编译和架构
    #####################################
    porj_list = os.listdir(f'/sdb/Version_recg/data/{config["name"]}/DBs/Embedding/')
    for proj in porj_list:
        if not _proj:
            pass
        elif proj != _proj:
            continue
        # 跨编译优化
        ###############################
        # DeBUG
        # if proj == 'cc1' or proj == 'libcrypto.so.3':
        #     continue
        ###############################
        if work_type == 'XO':

            opts_store = ['O0', 'O1', 'O2', 'O3']
            # opts_store = ['O2']
            opt_pair = []
            left_opt = 'O2'
            for right_opt in opts_store:
                opt_pair.append([left_opt, right_opt])
            for _opts in opt_pair:
                lib_dir = f'/sdb/Version_recg/data/{config["name"]}/DBs/sensitive_func/{proj}/X64/{_opts[0]}'
                target_dir = f'/sdb/Version_recg/data/{config["name"]}/DBs/Embedding/{proj}/X64/{_opts[1]}'
                archor_dir = f"/sdb/Version_recg/results/Archor_Version/{config['name']}/{proj}/X64/{_opts[0]}_{_opts[1]}"
                if not os.path.exists(target_dir):
                    print(f"target_dir: {target_dir} not exist")
                    continue
                if not os.path.exists(archor_dir):
                    os.makedirs(archor_dir)
                arch_ver_dict = dict()
                start_time = time.time()
                for tar_ver in os.listdir(target_dir):
                # for tar_ver in ['7.40']:
                    ############# DBG ##################
                    # if not tar_ver.startswith('3'):
                    #     continue
                    ####################################
                    target_file = f"{target_dir}/{tar_ver}/{proj}/all_func.pkl"
                    if not os.path.exists(lib_dir):
                        arch_ver_dict[tar_ver] = [[None, None]]
                        continue
                    
                    arch_ver = match_OSS(target_file, lib_dir)
                    
                    arch_ver_dict[tar_ver] = arch_ver
                print(f"{proj}  {_opts}  {(time.time() - start_time)/len(os.listdir(target_dir))}")
                archor_file = os.path.join(archor_dir, 'archor_ver.json')
                
                write_json(arch_ver_dict, archor_file)
                result_file = os.path.join(archor_dir, 'result.json')
                final_result = caculate_result(arch_ver_dict)
                write_json(final_result, result_file)
        ######################################
        # 跨架构
        if work_type == 'XA':
            _opts = ['X64', 'ARM']
            lib_dir = f'/sdb/Version_recg/data/{config["name"]}/DBs/sensitive_func/{proj}/{_opts[0]}/O2'
            target_dir = f'/sdb/Version_recg/data/{config["name"]}/DBs/Embedding/{proj}/{_opts[1]}/O2'
            archor_dir = f"/sdb/Version_recg/results/Archor_Version/{config['name']}/{proj}/{_opts[0]}_{_opts[1]}/O2"
            if not os.path.exists(archor_dir):
                    os.makedirs(archor_dir)
            arch_ver_dict = dict()

            for tar_ver in os.listdir(target_dir):
                target_file = f"/sdb/Version_recg/data/{config['name']}/DBs/Embedding/{proj}/{_opts[1]}/O2/{tar_ver}/{proj}/all_func.pkl"
                if not os.path.exists(lib_dir):
                        arch_ver_dict[tar_ver] = [[None, None]]
                        continue
                arch_ver = match_OSS(target_file, lib_dir)
                arch_ver_dict[tar_ver] = arch_ver
            archor_file = os.path.join(archor_dir, 'archor_ver.json')
            
            write_json(arch_ver_dict, archor_file)
            result_file = os.path.join(archor_dir, 'result.json')
            final_result = caculate_result(arch_ver_dict)
            write_json(final_result, result_file)
