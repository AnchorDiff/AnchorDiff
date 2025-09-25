import os
import pandas as pd
from pathlib import Path
import re

def analyze_libvdiff_results(base_path="results/Detail_match"):
    """
    分析LIBVDIFF测试结果，计算每个组件的平均正确率
    
    Args:
        base_path: 结果文件的基础路径
    """
    
    results = {}
    
    # 遍历OSS_lib和libvdiff两个主要目录
    for method_dir in ['OSS_lib', 'libvdiff']:
        method_path = Path(base_path) / method_dir
        
        if not method_path.exists():
            print(f"目录不存在: {method_path}")
            continue
            
        results[method_dir] = {}
        print(f"\n=== 分析 {method_dir} ===")
        
        # 遍历组件目录（如libblosc.so）
        for component_dir in method_path.iterdir():
            if not component_dir.is_dir():
                continue
                
            component_name = component_dir.name
            print(f"\n--- 组件: {component_name} ---")
            
            # 查找X64架构目录
            x64_path = component_dir / "X64"
            if not x64_path.exists():
                print(f"  未找到X64目录: {x64_path}")
                continue
            
            # 收集所有O2_O*目录（跳过O2_O2）
            opt_accuracies = []
            
            for opt_dir in x64_path.iterdir():
                if not opt_dir.is_dir():
                    continue
                
                # 匹配O2_O*模式，但跳过O2_O2
                if re.match(r'O2_O[013]$', opt_dir.name):
                    result_file = opt_dir / "rasult.csv"
                    
                    if not result_file.exists():
                        print(f"    {opt_dir.name}: 未找到rasult.csv文件")
                        continue
                    
                    try:
                        # 读取CSV文件
                        df = pd.read_csv(result_file)
                        
                        # 检查是否有result列
                        if 'result' not in df.columns:
                            print(f"    {opt_dir.name}: CSV文件中未找到result列")
                            continue
                        
                        # 计算True的比例
                        if len(df) > 0:
                            accuracy = (df['result'] == True).sum() / len(df)
                            opt_accuracies.append(accuracy)
                            print(f"    {opt_dir.name}: {accuracy:.4f} ({(df['result'] == True).sum()}/{len(df)})")
                        else:
                            print(f"    {opt_dir.name}: CSV文件为空")
                            
                    except Exception as e:
                        print(f"    {opt_dir.name}: 读取文件出错 - {e}")
                        continue
            
            # 计算该组件的平均正确率
            if opt_accuracies:
                avg_accuracy = sum(opt_accuracies) / len(opt_accuracies)
                results[method_dir][component_name] = {
                    'individual_accuracies': opt_accuracies,
                    'average_accuracy': avg_accuracy,
                    'count': len(opt_accuracies)
                }
                print(f"  >> 平均正确率: {avg_accuracy:.4f}")
            else:
                print(f"  >> 无有效数据")
                results[method_dir][component_name] = {
                    'individual_accuracies': [],
                    'average_accuracy': 0.0,
                    'count': 0
                }
    
    return results

def print_summary(results):
    """打印汇总结果"""
    print("\n" + "="*60)
    print("汇总结果")
    print("="*60)
    
    for method_name, method_results in results.items():
        print(f"\n{method_name}:")
        print("-" * 40)
        
        if not method_results:
            print("  无数据")
            continue
        
        # 按平均正确率排序
        sorted_components = sorted(
            method_results.items(), 
            key=lambda x: x[1]['average_accuracy'], 
            reverse=True
        )
        
        total_avg = []
        for component_name, data in sorted_components:
            if data['count'] > 0:
                print(f"  {component_name:<20}: {data['average_accuracy']:.4f} (基于{data['count']}个测试)")
                total_avg.append(data['average_accuracy'])
        
        if total_avg:
            overall_avg = sum(total_avg) / len(total_avg)
            print(f"  {'总体平均正确率':<20}: {overall_avg:.4f}")

def save_results_to_csv(results, output_file="libvdiff_analysis_summary.csv"):
    """将结果保存到CSV文件"""
    rows = []
    
    for method_name, method_results in results.items():
        for component_name, data in method_results.items():
            rows.append({
                'Method': method_name,
                'Component': component_name,
                'Average_Accuracy': data['average_accuracy'],
                'Test_Count': data['count'],
                'Individual_Accuracies': ','.join(map(str, data['individual_accuracies']))
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    # 分析结果
    results = analyze_libvdiff_results()
    
    # 打印汇总
    print_summary(results)
    
    # 保存到CSV
    # save_results_to_csv(results)
