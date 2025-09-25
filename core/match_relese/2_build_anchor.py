import os
import pickle
import torch
import itertools
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import copy
import json
# 比较两个相邻版本的差异
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
config = read_json('./config.json')['dataset']

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

def save_pickle_file(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

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

def match_bt_versions(cur_data, next_data):
    # 比较cur_data中存在而next_data中被删除掉的函数
    cur_names = cur_data['name']
    next_names = next_data['name']

    deleted_names = list(set(cur_names) - set(next_names))
    # 删除deleted_names中以"output_"开头的元素
    deleted_names = [name for name in deleted_names if not (name.startswith("sub_") or name.startswith("output_") or name.startswith("gen_split_") or name.startswith("gen_peephole2_") or name.startswith("pattern"))]

    deleted_embs = torch.stack([cur_data['embedding '][cur_names.index(name)] for name in deleted_names])
    deleted_imm_store = [cur_data['imm_store'][cur_names.index(name)] for name in deleted_names]
    
    
    # 比较cur_data中不存在而next_data中新增的函数
    added_names = list(set(next_names) - set(cur_names))
    added_names = [name for name in added_names if not (name.startswith("sub_") or name.startswith("output_") or name.startswith("gen_split_") or name.startswith("gen_peephole2_") or name.startswith("pattern"))]

    added_embs = torch.stack([next_data['embedding '][next_names.index(name)] for name in added_names])
    added_imm_store = [next_data['imm_store'][next_names.index(name)] for name in added_names]
    
    
    # 比较同名函数的embedding差异
    common_names = list(set(cur_names) & set(next_names))
    result = {}
    ##################################
    # 使用矩阵运算计算相似度
    cur_embeddings = torch.stack([cur_data['embedding '][cur_names.index(name)] for name in common_names])
    next_embeddings = torch.stack([next_data['embedding '][next_names.index(name)] for name in common_names])

    # 归一化向量
    cur_embeddings_normalized = torch.nn.functional.normalize(cur_embeddings, p=2, dim=1)
    next_embeddings_normalized = torch.nn.functional.normalize(next_embeddings, p=2, dim=1)

    # 计算余弦相似度
    similarities = torch.sum(cur_embeddings_normalized * next_embeddings_normalized, dim=1).tolist()

    # 将结果存入字典
    result = {name: similarity for name, similarity in zip(common_names, similarities)}
    #####################################
    # for name in common_names:
    #     cur_embedding = cur_data['embedding '][cur_names.index(name)]
    #     next_embedding = next_data['embedding '][next_names.index(name)]
    #     similarity = cosine_similarity(cur_embedding.unsqueeze(0), next_embedding.unsqueeze(0)).item()
    #     result[name] = similarity

    threshold = 0.8
    changed_names = []

    for name, similarity in result.items():
        if similarity >= threshold or name.startswith("sub_") or name.startswith("output_") or name.startswith("gen_split_") or name.startswith("gen_peephole2_") or name.startswith("pattern"):
            pass
        else:
            changed_names.append(name)
    if len(changed_names) == 0:
        changed_embs = None
        changed_imm_store = []
    else:
        changed_embs = torch.stack([cur_data['embedding '][cur_names.index(name)] for name in changed_names])
        changed_imm_store = [cur_data['imm_store'][cur_names.index(name)] for name in changed_names]
    
    
    ret_names = {
        'deleted': deleted_names,
        'added': added_names,
        'changed': changed_names
    }
    ret_embs = {
        'deleted': deleted_embs,
        'added': added_embs,
        'changed': changed_embs
    }
    ret_imm_store = {
        'deleted': deleted_imm_store,
        'added': added_imm_store,
        'changed': changed_imm_store
    }
    
    return  ret_names, ret_embs, ret_imm_store

    

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
    count_above_threshold = (max_values > 0.6).sum().item()
    
    return {
        'ratio': count_above_threshold / len(lib_names),
        "mean": avg_similarity,
        "num": count_above_threshold ,
        'total': len(target_names),
        'total_lib': len(lib_names),
        
    }
    
def update_sensitive_func(sensitive_func, senstive_func_store):
    for _type, func_set in sensitive_func.items():
        senstive_func_store[_type].update(func_set)
    return senstive_func_store

def print_lib_num(sensitive_func):
    for ver, lib in sensitive_func.items():
        _type_list = ['deleted', 'added', 'changed']
        print('\n')
        for _type in _type_list:
            print(f'{ver} {_type} num: {len(lib[_type])}')
            
def del_funcs(args):
    sensitive_func, sensitive_embs, sensitive_immstore, cur_file_data, next_file_data, cur_ver, next_ver = args
    # 对于cur_ver, 其deleted的函数在next_ver中都不允许出现
    # 对于next_ver, 其added的函数在cur_ver中都不允许出现
    cur_names = cur_file_data['name']
    next_names = next_file_data['name']

    # 删除cur_sensitive_func中在next_names中出现的函数,以及相同索引下的embedding和immstore
    if cur_ver in sensitive_func.keys():
        # 备份cur_sensitive_func
        _cur_sensitive_func_del = copy.deepcopy(sensitive_func[cur_ver]['deleted'])
        _cur_sensitive_func_add = copy.deepcopy(sensitive_func[cur_ver]['added'])
        
        
        cur_sensitive_func = sensitive_func[cur_ver]
        cur_sensitive_func['deleted'] = [name for name in cur_sensitive_func['deleted'] if name not in next_names]
        cur_sensitive_func['added'] = [name for name in cur_sensitive_func['added'] if name in next_names]
        
        cur_sensitive_embs = sensitive_embs[cur_ver]
        cur_sensitive_immstore = sensitive_immstore[cur_ver]
        cur_sensitive_embs['deleted'] = [cur_sensitive_embs['deleted'][i] for i, name in enumerate(_cur_sensitive_func_del) if name not in next_names]
        cur_sensitive_immstore['deleted'] = [cur_sensitive_immstore['deleted'][i] for i, name in enumerate(_cur_sensitive_func_del) if name not in next_names]
        
        cur_sensitive_embs['added'] = [cur_sensitive_embs['added'][i] for i, name in enumerate(_cur_sensitive_func_add) if name in next_names]
        cur_sensitive_immstore['added'] = [cur_sensitive_immstore['added'][i] for i, name in enumerate(_cur_sensitive_func_add) if name in next_names]
    # 删除next_sensitive_func中在cur_names中出现的函数,以及相同索引下的embedding和immstore
    if next_ver in sensitive_func.keys():
        next_sensitive_func = sensitive_func[next_ver]
        # 备份next_sensitive_func
        _next_sensitive_func_del = copy.deepcopy(sensitive_func[next_ver]['deleted'])
        _next_sensitive_func_add = copy.deepcopy(sensitive_func[next_ver]['added'])
        next_sensitive_func['added'] = [name for name in next_sensitive_func['added'] if name not in cur_names]
        next_sensitive_func['deleted'] = [name for name in next_sensitive_func['deleted'] if name in cur_names]
        
        next_sensitive_embs = sensitive_embs[next_ver]
        next_sensitive_immstore = sensitive_immstore[next_ver]
        next_sensitive_embs['added'] = [next_sensitive_embs['added'][i] for i, name in enumerate(_next_sensitive_func_add) if name not in cur_names]
        next_sensitive_immstore['added'] = [next_sensitive_immstore['added'][i] for i, name in enumerate(_next_sensitive_func_add) if name not in cur_names]
        
        next_sensitive_embs['deleted'] = [next_sensitive_embs['deleted'][i] for i, name in enumerate(_next_sensitive_func_del) if name in cur_names]
        next_sensitive_immstore['deleted'] = [next_sensitive_immstore['deleted'][i] for i, name in enumerate(_next_sensitive_func_del) if name in cur_names]
            
def match_archor(lib_dir, proj):
    # archor_pairs_dict = {
    #     'cc1' : [
    #         ('7.5.0', '8.1.0'), ('8.5.0', '9.1.0'), ('9.5.0', '10.1.0'), ('10.5.0', '11.1.0'), 
    #         ('11.5.0', '12.1.0'), ('12.4.0', '13.1.0'), ( '13.3.0', '14.1.0')],
    #     'libcrypto.so.3' : [('1.0.0.3', '1.0.1.1'), ('1.0.1.4', '1.0.2'), ('1.0.2.4', '3.0.0'), ('3.0.16', '3.1.0'), ('3.1.8', '3.2.0'), ('3.3.3', '3.4.0')],
    #     'libcrypto.so.3-s' : [('3.0.16', '3.1.0'), ('3.1.8', '3.2.0'), ('3.3.3', '3.4.0'),],
    #     'libfreetype.so'    : [('2.4.11', '2.4.12'), ('2.5.5', '2.7.1'), ('2.12.1', '2.13.0')],
    #     }

    # proj = 'libcrypto.so'
    Archor_path = os.path.join(lib_dir.replace('Embedding', 'Archor'), proj, f'archor.pkl')
    archor_data = read_pickle_file(Archor_path)
    
    archor_pairs = archor_data
    if len(archor_pairs) == 0:
        print('archor_pairs is empty')
        return
    def version_key(version):
        return [int(part) for part in version.split('.')]
    proj_path = os.path.join(lib_dir, proj)
    arch = 'X64'
    arch_path = os.path.join(proj_path, arch)
    if not os.path.exists(arch_path):
        arch_path = os.path.join(proj_path, 'x64')
    opt = 'O2'
    opt_path = os.path.join(arch_path, opt)
    versions = os.listdir(opt_path)
    versions = sorted(versions, key=version_key)
    sensitive_func = {}
    sensitive_embs = {}
    sensitive_immstore = {}
    for key_pair in archor_pairs:
        sensitive_func[key_pair[0]] = {
                    'deleted': list(),
                    'added': list(),
                    'changed': list()
                }
        cur_file_path = os.path.join(opt_path, key_pair[0], os.listdir(os.path.join(opt_path, key_pair[0]))[0],'all_func.pkl')
        next_file_path = os.path.join(opt_path, key_pair[1], os.listdir(os.path.join(opt_path, key_pair[1]))[0], 'all_func.pkl')
        cur_file_data = read_pickle_file(cur_file_path)
        next_file_data = read_pickle_file(next_file_path)
        try:
            ret_names, ret_embs, ret_immstore = match_bt_versions(cur_file_data, next_file_data)
        except Exception as e:
            print(f"Error matching versions {key_pair[0]} and {key_pair[1]}: {e}")
            ret_names, ret_embs, ret_immstore = match_bt_versions(cur_file_data, next_file_data)
            continue
            
        sensitive_func[key_pair[0]] = ret_names
        sensitive_embs[key_pair[0]] = ret_embs
        sensitive_immstore[key_pair[0]] = ret_immstore
        
        
    # # 全版本搜索，剔除增后删除和删后增加的函数
    # ###################################################
    # # versions = [version for version in versions if version.startswith('1')]
    # ###################################################

    # version_pairs = list(itertools.combinations(versions, 2))
    # for key_pair in version_pairs:
    #     # pbar.set_description(f"{proj}/{arch}/{opt}/{key_pair[0]}-{key_pair[1]}")
    #     cur_file_path = os.path.join(opt_path, key_pair[0], os.listdir(os.path.join(opt_path, key_pair[0]))[0],'all_func.pkl')
    #     next_file_path = os.path.join(opt_path, key_pair[1], os.listdir(os.path.join(opt_path, key_pair[1]))[0], 'all_func.pkl')
    #     try:
    #         cur_file_data = read_pickle_file(cur_file_path)
    #         next_file_data = read_pickle_file(next_file_path)
    #         next_file_list = [next_file_data]
    #     except:
    #         continue
    #     next_file_path_1 = next_file_path.replace('O2', 'O0')
    #     next_file_path_2 = next_file_path.replace('O2', 'O1')
    #     if os.path.isfile(next_file_path_1):
    #         next_file_data_1 = read_pickle_file(next_file_path_1)
    #         next_file_list.append(next_file_data_1)
    #     if os.path.isfile(next_file_path_2):
    #         next_file_data_2 = read_pickle_file(next_file_path_2)
    #         next_file_list.append(next_file_data_2)
            
    #     for _next_file_data in next_file_list:
    #         del_funcs((sensitive_func, sensitive_embs, sensitive_immstore, cur_file_data, _next_file_data, key_pair[0], key_pair[1]))

        # del_funcs((sensitive_func, sensitive_embs, sensitive_immstore, cur_file_data, next_file_data, key_pair[0], key_pair[1]))
    print('*'*10)
    print_lib_num(sensitive_func)

    # 保存敏感函数
    savedir = os.path.join(os.path.dirname(lib_dir), 'sensitive_func', proj, arch, opt)
    if not os.path.exists(savedir):
            os.makedirs(savedir)
    for key, value in sensitive_func.items():
        save_pickle_file(os.path.join(savedir, key + '.pkl'), value)
    # 保存敏感函数的embedding
    savedir = os.path.join(os.path.dirname(lib_dir), 'sensitive_func', proj, arch, opt)
    if not os.path.exists(savedir):
            os.makedirs(savedir)
    for key, value in sensitive_embs.items():
        for _type, emb in value.items():
            if emb == None:
                continue
            if len(emb) == 0 or type(emb) == torch.Tensor:
                continue
            # 取出embedding
            emb = torch.stack(emb)
            value[_type] = emb
        save_pickle_file(os.path.join(savedir, key + '_emb.pkl'), value)
        
    # 保存敏感函数的immstore
    savedir = os.path.join(os.path.dirname(lib_dir), 'sensitive_func', proj, arch, opt)
    if not os.path.exists(savedir):
            os.makedirs(savedir)
    for key, value in sensitive_immstore.items():
        save_pickle_file(os.path.join(savedir, key + '_immstore.pkl'), value)

if __name__ == '__main__':
    lib_dir = os.path.join(config['root'], config['name'], 'DBs', 'Embedding')   #'data/OSS_lib/DBs/Embedding'
    _proj = input('proj: ')
    for proj in os.listdir(lib_dir):
        ################
        # if proj == 'cc1':
        #     continue
        ################
        if not _proj:
            pass
        elif not proj == _proj:
            continue
        
        match_archor(lib_dir, proj)
        #print(os.path.join(lib_dir, proj))
    # match_archor(lib_dir)
    #print(file_dict)