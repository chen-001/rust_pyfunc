import numpy as np
import rust_pyfunc

# 简单测试 - 使用numpy数组和时间戳格式的日期
dates = np.array([1640995200, 1640995200, 1640995200, 1641081600, 1641081600], dtype=np.int64)  # 时间戳格式
factors = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

print("测试数据:")
print("dates:", dates)
print("factors:", factors)
print("dates类型:", type(dates), dates.dtype)
print("factors类型:", type(factors), factors.dtype)

try:
    if hasattr(rust_pyfunc, 'factor_grouping'):
        print("✓ factor_grouping函数已找到")
        result = rust_pyfunc.factor_grouping(dates, factors, 3)
        print("Result:", result)
        print("类型:", type(result))
        print("形状:", result.shape if hasattr(result, 'shape') else 'No shape')
        print("数据类型:", result.dtype if hasattr(result, 'dtype') else 'No dtype')
        
        # 验证结果
        print("\n结果验证:")
        for i, (d, f, g) in enumerate(zip(dates, factors, result)):
            print(f"索引{i}: 日期={d}, 因子={f}, 分组={g}")
            
        # 检查分组逻辑
        print("\n分组逻辑检查:")
        unique_dates = np.unique(dates)
        for date in unique_dates:
            mask = dates == date
            date_factors = factors[mask]
            date_groups = result[mask]
            sorted_indices = np.argsort(date_factors)
            print(f"日期 {date}:")
            print(f"  因子值: {date_factors[sorted_indices]}")
            print(f"  分组号: {date_groups[sorted_indices]}")
            
    else:
        print("✗ factor_grouping函数未找到")
        print("可用函数:", [f for f in dir(rust_pyfunc) if not f.startswith('__')][:10])
        
except Exception as e:
    print("错误:", e)
    import traceback
    traceback.print_exc()