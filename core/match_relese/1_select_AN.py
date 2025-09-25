import os
import pickle
import json
import torch
import itertools
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
# 比较两个相邻版本的差异
# 修改比较策略，对于delete。在cur_ver之后都不允许出现
# 对于add，在cur_ver之前都不允许出现

# 新增修订：ADD库中函数，未来不能被删除，del库中之前必须存在
# 功能修改：获取目标组件的archor

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

config = read_json('./config.json')['dataset']

def version_key(version):
        return [int(part) for part in version.split('.')]

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
    cur_sensitive_func = {
        "deleted": set(),
        "added": set(),
        "changed": set()
    }
    next_sensitive_func = {
        "deleted": set(),
        "added": set(),
        "changed": set()
    }
    deleted_names = list(set(cur_names) - set(next_names))
    # 删除deleted_names中以"sub_"开头的元素
    deleted_names = [name for name in deleted_names if not (name.startswith("sub_") or name.startswith("output_") or name.startswith("gen_split_") or name.startswith("gen_peephole2_") or name.startswith("pattern"))]
    cur_sensitive_func['deleted'].update(deleted_names)
    
    
    # 比较cur_data中不存在而next_data中新增的函数
    added_names = list(set(next_names) - set(cur_names))
    added_names = [name for name in added_names if not (name.startswith("sub_") or name.startswith("output_") or name.startswith("gen_split_") or name.startswith("gen_peephole2_") or name.startswith("pattern"))]
    next_sensitive_func['added'].update(added_names)
    
    
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

    threshold = 0.6

    for name, similarity in result.items():
        if similarity >= threshold:
            pass
        else:
            # cur_sensitive_func.add(name)
            # next_sensitive_func.add(name)
            cur_sensitive_func['changed'].add(name)
            next_sensitive_func['changed'].add(name)
            

    return  cur_sensitive_func, next_sensitive_func

    

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
    sensitive_func, cur_file_data, next_file_data, cur_ver, next_ver = args
    # 对于cur_ver, 其deleted的函数在next_ver中都不允许出现
    # 对于next_ver, 其added的函数在cur_ver中都不允许出现
    cur_names = cur_file_data['name']
    next_names = next_file_data['name']
    cur_sensitive_func = sensitive_func[cur_ver]
    next_sensitive_func = sensitive_func[next_ver]
    # 删除del中在next_names中出现的函数
    cur_sensitive_func['deleted'] = [name for name in cur_sensitive_func['deleted'] if name not in next_names]
    cur_sensitive_func['added'] = [name for name in cur_sensitive_func['added'] if name in next_names]
    
    
    # 删除next_sensitive_func中在cur_names中出现的函数
    next_sensitive_func['added'] = [name for name in next_sensitive_func['added'] if name not in cur_names]
    next_sensitive_func['deleted'] = [name for name in next_sensitive_func['deleted'] if name in cur_names]
    
def get_archor(sensitive_func, proj_num):
    # 通过比较各个版本的deleted和added函数，获取archor
    versions = list(sensitive_func.keys())
    archor = []
    for idx, ver in enumerate(versions):
        if idx < 2 or idx >= len(versions) - 1:
            continue
        prev_ver = versions[idx - 1]
        prev_sensitive_func = sensitive_func[prev_ver]
        cur_sensitive_func = sensitive_func[ver]
        # 1 当前版本的deleted数量大于proj_num，且added数量也大于proj_num
        # if len(cur_sensitive_func['deleted']) > proj_num and len(cur_sensitive_func['added']) > proj_num:
        #     archor.append((ver, ver))
        #     cur_sensitive_func['deleted'] = []
        # 2 当前版本的deleted数量小于proj_num，但是added数量大于proj_num且prev_ver的deleted数量大于proj_num
        if len(cur_sensitive_func['deleted']) < proj_num and len(cur_sensitive_func['added']) > proj_num and len(prev_sensitive_func['deleted']) > proj_num:
            archor.append((prev_ver, ver))
            cur_sensitive_func['deleted'] = []
    return archor

def get_archor_map(archors, versions):
    # 获取由archor node 分割而成的区域，以及区域内的版本号
    archor_map = {}
    if len(archors) == 0:
        archor_map[(None, None)] = versions
        return archor_map
    if len(archors) == 1:
        archor_map[(None, archors[0][0])] = []
        archor_map[(archors[0][0], None)] = []
        for version in versions:
            if version_key(version) <= version_key(archors[0][0]):
                archor_map[(None, archors[0][0])].append(version)
            elif version_key(version) >= version_key(archors[0][1]):
                archor_map[(archors[0][0], None)].append(version)
        return archor_map
    for idx, archor in enumerate(archors):
        if idx == 0:
            archor_map[(None, archor[0])] = []
            archor_map[(archor[0], archors[idx + 1][0])] = []
            for version in versions:
                if version_key(version) <= version_key(archor[0]):
                    archor_map[(None, archor[0])].append(version)
                elif version_key(version) >= version_key(archor[1]) and version_key(version) <= version_key(archors[idx + 1][0]):
                    archor_map[(archor[0], archors[idx + 1][0])].append(version)
        elif idx == len(archors) - 1:
            archor_map[(archor[0], None)] = []
            for version in versions:
                if version_key(version) >= version_key(archor[1]):
                    archor_map[(archor[0], None)].append(version)
        else:
            archor_map[(archor[0], archors[idx + 1][0])] = []
            for version in versions:
                if version_key(version) >= version_key(archor[1]) and version_key(version) <= version_key(archors[idx + 1][0]):
                    archor_map[(archor[0], archors[idx + 1][0])].append(version)
    return archor_map


def match_all(lib_dir):
    project_list = os.listdir(lib_dir)
    ##############################################
    # project_list = ['sqlite3'] # 'libfreetype.so' 'libfreetype.so' 'libaws-c-common.so.1.0.0'
    
    ###############################################
    
    pbar = tqdm(total=len(project_list), ncols=100)
    _proj = input('proj:')
    for proj in project_list:
        ##############################
        # if proj == 'cc1':
        #     continue
        ##############################
        if not _proj:
            pass
        elif  proj != _proj:
            continue
        proj_path = os.path.join(lib_dir, proj)
        pbar.update(1)
        print('\n'+'#'*30)
        print(proj)
        proj_num = 5
        for arch in os.listdir(proj_path):
            arch_path = os.path.join(proj_path, arch)
            
            if not ('x64' in arch or 'X64' in arch):
                    continue

            for opt in os.listdir(arch_path):
                ######################################
                if not 'O2' in opt:
                    continue
                
                ######################################
                opt_path = os.path.join(arch_path, opt)
                versions = os.listdir(opt_path)
                try:
                    versions = sorted(versions, key=version_key)
                except:
                    print(versions)
                    continue
                
                # 1 相邻版本匹配，获取敏感函数
                sensitive_func = {}
                version_pairs = [(versions[i], versions[i + 1]) for i in range(len(versions) - 1)]
                for key_pair in version_pairs:
                    pbar.set_description(f"{proj}/{arch}/{opt}/{key_pair[0]}-{key_pair[1]}")
                    
                    if not key_pair[0] in sensitive_func:
                        sensitive_func[key_pair[0]] = {
                            'deleted': set(),
                            'added': set(),
                            'changed': set()
                        }
                    if not key_pair[1] in sensitive_func:
                        sensitive_func[key_pair[1]] = {
                            'deleted': set(),
                            'added': set(),
                            'changed': set()
                        }
                    cur_file_path = os.path.join(opt_path, key_pair[0], os.listdir(os.path.join(opt_path, key_pair[0]))[0],'all_func.pkl')
                    next_file_path = os.path.join(opt_path, key_pair[1], os.listdir(os.path.join(opt_path, key_pair[1]))[0], 'all_func.pkl')
                    try:
                        cur_file_data = read_pickle_file(cur_file_path)
                        next_file_data = read_pickle_file(next_file_path)
                    except:
                        continue
                    proj_num = max(proj_num, len(cur_file_data['name'])//150)
                    cur_sensitive_func, next_sensitive_func = match_bt_versions(cur_file_data, next_file_data)
                    sensitive_func[key_pair[0]] = update_sensitive_func(cur_sensitive_func, sensitive_func[key_pair[0]])
                    sensitive_func[key_pair[1]] = update_sensitive_func(next_sensitive_func, sensitive_func[key_pair[1]])
                # print_lib_num(sensitive_func)
                
                # # 2 全版本搜索，剔除增后删除和删后增加的函数

                # version_pairs = list(itertools.combinations(versions, 2))
                # for key_pair in version_pairs:
                #     next_file_list = []
                #     pbar.set_description(f"{proj}/{arch}/{opt}/{key_pair[0]}-{key_pair[1]}")
                #     cur_file_path = os.path.join(opt_path, key_pair[0], os.listdir(os.path.join(opt_path, key_pair[0]))[0],'all_func.pkl')
                #     next_file_path = os.path.join(opt_path, key_pair[1], os.listdir(os.path.join(opt_path, key_pair[1]))[0], 'all_func.pkl')
                #     try:
                #         cur_file_data = read_pickle_file(cur_file_path)
                #         next_file_data = read_pickle_file(next_file_path)
                #         next_file_list.append(next_file_data)
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
                #         del_funcs((sensitive_func, cur_file_data, _next_file_data, key_pair[0], key_pair[1]))
                # print_lib_num(sensitive_func)
                archor = get_archor(sensitive_func, proj_num)
                archor_map = get_archor_map(archor, versions)
                print('\narchor:', archor)
                
                # Save archor and archor_map
                
                savedir = os.path.join(os.path.dirname(lib_dir), 'Archor', proj)
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                archorpath = os.path.join(savedir, 'archor.pkl')
                map_path = os.path.join(savedir, 'archor_map.pkl')
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                with open(archorpath, 'wb') as f:
                    pickle.dump(archor, f)
                with open(map_path, 'wb') as f:
                    pickle.dump(archor_map, f)

    pbar.close()
    




if __name__ == '__main__':
    lib_dir = os.path.join(config['root'], config['name'], 'DBs', 'Embedding')
    # lib_dir = 'data/OSS_lib/DBs/Embedding'
    match_all(lib_dir)
    #print(file_dict)