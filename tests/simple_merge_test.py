"""
简单的fast_merge功能测试
"""

import numpy as np
import pandas as pd
import rust_pyfunc as rp

# 测试基础内连接
print("=== 测试基础内连接 ===")

left_data = np.array([
    [1.0, 100.0],
    [2.0, 200.0],
    [3.0, 300.0],
], dtype=np.float64)

right_data = np.array([
    [1.0, 10.0],
    [2.0, 20.0],
    [4.0, 40.0],
], dtype=np.float64)

print("左表数据:")
print(left_data)
print("右表数据:")
print(right_data)

# 使用fast_merge
try:
    indices, merged_data = rp.fast_merge(
        left_data, right_data,
        left_keys=[0], right_keys=[0],
        how="inner"
    )
    
    print(f"\nfast_merge结果（{len(merged_data)}行）:")
    for i, row in enumerate(merged_data):
        print(f"行{i}: {row}")
    
    # 验证基本逻辑
    assert len(merged_data) == 2, f"内连接应该有2行，实际{len(merged_data)}行"
    
    # 检查第一行 (key=1)
    assert merged_data[0][0] == 1.0, "第一行左表key错误"
    assert merged_data[0][1] == 100.0, "第一行左表value错误" 
    assert merged_data[0][2] == 1.0, "第一行右表key错误"
    assert merged_data[0][3] == 10.0, "第一行右表value错误"
    
    # 检查第二行 (key=2)
    assert merged_data[1][0] == 2.0, "第二行左表key错误"
    assert merged_data[1][1] == 200.0, "第二行左表value错误"
    assert merged_data[1][2] == 2.0, "第二行右表key错误"
    assert merged_data[1][3] == 20.0, "第二行右表value错误"
    
    print("✓ 内连接测试通过")
    
except Exception as e:
    print(f"✗ 内连接测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试左连接
print("\n=== 测试左连接 ===")

try:
    indices, merged_data = rp.fast_merge(
        left_data, right_data,
        left_keys=[0], right_keys=[0],
        how="left"
    )
    
    print(f"\nfast_merge左连接结果（{len(merged_data)}行）:")
    for i, row in enumerate(merged_data):
        print(f"行{i}: {row}")
    
    # 验证左连接逻辑
    assert len(merged_data) == 3, f"左连接应该有3行，实际{len(merged_data)}行"
    
    # 检查第三行应该有NaN (key=3在右表中不存在)
    assert merged_data[2][0] == 3.0, "第三行左表key错误"
    assert merged_data[2][1] == 300.0, "第三行左表value错误"
    assert np.isnan(merged_data[2][2]), "第三行右表key应该是NaN"
    assert np.isnan(merged_data[2][3]), "第三行右表value应该是NaN"
    
    print("✓ 左连接测试通过")
    
except Exception as e:
    print(f"✗ 左连接测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 所有基础测试完成 ===")