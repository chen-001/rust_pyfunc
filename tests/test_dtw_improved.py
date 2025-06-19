import numpy as np
import time
import sys
sys.path.append('/home/chenzongwei/rustcode/rust_pyfunc/python')
from rust_pyfunc import fast_dtw_distance, dtw_distance

def test_dtw_improvements():
    """测试fast_dtw_distance函数改进后的性能和正确性"""
    print("开始测试改进后的fast_dtw_distance函数...\n")
    
    # 测试1：简单序列，不同radius设置下的结果
    print("测试1：简单序列，不同radius设置下的结果")
    s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    
    # 不使用radius约束
    result_no_radius = fast_dtw_distance(s1, s2)
    print(f"不使用radius约束: {result_no_radius:.6f}")
    
    # 使用合理的radius
    result_radius_2 = fast_dtw_distance(s1, s2, radius=2)
    print(f"使用radius=2: {result_radius_2:.6f}")
    
    # 使用非常小的radius，测试自动调整功能
    result_radius_1 = fast_dtw_distance(s1, s2, radius=1)
    print(f"使用radius=1 (应该自动调整): {result_radius_1:.6f}")
    
    # 验证结果是否一致
    print(f"三种方法结果是否一致: {np.isclose(result_no_radius, result_radius_2) and np.isclose(result_no_radius, result_radius_1)}")
    print()
    
    # 测试2：相差较大的序列，测试自动调整radius
    print("测试2：相差较大的序列，测试自动调整radius")
    s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    s2 = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])  # 完全相反的序列
    
    # 不使用radius约束
    result_no_radius = fast_dtw_distance(s1, s2)
    print(f"不使用radius约束: {result_no_radius:.6f}")
    
    # 使用不同大小的radius
    for r in [1, 2, 3, 5]:
        result = fast_dtw_distance(s1, s2, radius=r)
        print(f"使用radius={r}: {result:.6f}, 是否无穷大: {np.isinf(result)}")
    
    print()
    
    # 测试3：性能测试
    print("测试3：性能测试")
    # 创建较长的序列
    n = 500
    s1 = np.sin(np.linspace(0, 10*np.pi, n))
    s2 = np.sin(np.linspace(0.5, 10.5*np.pi, n))
    
    # 测试原始dtw_distance
    start = time.time()
    result1 = dtw_distance(s1, s2)
    time1 = time.time() - start
    
    # 测试改进后的fast_dtw_distance（不使用radius）
    start = time.time()
    result2 = fast_dtw_distance(s1, s2)
    time2 = time.time() - start
    
    # 测试改进后的fast_dtw_distance（使用radius）
    start = time.time()
    result3 = fast_dtw_distance(s1, s2, radius=20)
    time3 = time.time() - start
    
    print(f"原始DTW距离: {result1:.6f}, 耗时: {time1:.6f}秒")
    print(f"快速DTW距离(无radius): {result2:.6f}, 耗时: {time2:.6f}秒, 加速比: {time1/time2:.2f}倍")
    print(f"快速DTW距离(radius=20): {result3:.6f}, 耗时: {time3:.6f}秒, 加速比: {time1/time3:.2f}倍")
    print(f"结果误差: {abs(result1 - result2):.8f}, {abs(result1 - result3):.8f}")
    
    # 测试4：空序列处理测试
    print("\n测试4：空序列处理测试")
    empty_seq = np.array([])
    normal_seq = np.array([1.0, 2.0, 3.0])
    
    # 测试空序列与正常序列
    result_empty1 = fast_dtw_distance(empty_seq, normal_seq)
    result_empty2 = fast_dtw_distance(normal_seq, empty_seq)
    
    # 测试两个空序列
    result_empty3 = fast_dtw_distance(empty_seq, empty_seq)
    
    print(f"空序列与正常序列(空在前): 结果: {result_empty1}, 是否为NaN: {np.isnan(result_empty1)}")
    print(f"空序列与正常序列(空在后): 结果: {result_empty2}, 是否为NaN: {np.isnan(result_empty2)}")
    print(f"两个空序列: 结果: {result_empty3}, 是否为NaN: {np.isnan(result_empty3)}")
    
    print("\n所有测试完成！")

if __name__ == "__main__":
    test_dtw_improvements()
