"""
验证Timestamp类型错误修复
重现原始错误并验证修复效果
"""

import pandas as pd
import numpy as np
import rust_pyfunc as rp


def reproduce_original_error():
    """重现原始的Timestamp类型错误"""
    print("=== 重现原始Timestamp错误 ===")
    
    # 创建包含Timestamp类型的DataFrame
    df1 = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'value1': [100, 200, 300]
    })
    
    df2 = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-04']),
        'value2': [10, 20, 40]
    })
    
    print("数据1:")
    print(df1)
    print(f"date列类型: {df1['date'].dtype}")
    
    print("\n数据2:")
    print(df2)
    print(f"date列类型: {df2['date'].dtype}")
    
    try:
        print("\n尝试使用rust_pyfunc进行Timestamp键合并...")
        result = rp.fast_inner_join_df(df1, df2, on='date')
        print(f"✅ 成功！合并结果: {len(result)}行")
        print(result)
        return True
        
    except ValueError as e:
        if "不支持的键类型: Timestamp" in str(e):
            print(f"❌ 原始错误重现: {e}")
            return False
        else:
            print(f"❌ 其他错误: {e}")
            return False
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        return False


def test_various_datetime_types():
    """测试各种日期时间类型"""
    print("\n=== 测试各种日期时间类型 ===")
    
    from datetime import date, datetime
    
    test_cases = [
        {
            'name': 'pandas.Timestamp',
            'data1': pd.DataFrame({
                'key': pd.to_datetime(['2024-01-01', '2024-01-02']),
                'val': [1, 2]
            }),
            'data2': pd.DataFrame({
                'key': pd.to_datetime(['2024-01-01', '2024-01-03']),
                'val': [10, 30]
            })
        },
        {
            'name': 'datetime.date',
            'data1': pd.DataFrame({
                'key': [date(2024, 1, 1), date(2024, 1, 2)],
                'val': [1, 2]
            }),
            'data2': pd.DataFrame({
                'key': [date(2024, 1, 1), date(2024, 1, 3)],
                'val': [10, 30]
            })
        },
        {
            'name': 'datetime.datetime',
            'data1': pd.DataFrame({
                'key': [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 2, 11, 0)],
                'val': [1, 2]
            }),
            'data2': pd.DataFrame({
                'key': [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 3, 12, 0)],
                'val': [10, 30]
            })
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n测试 {case['name']}:")
        try:
            result = rp.fast_inner_join_df(case['data1'], case['data2'], on='key')
            print(f"  ✅ 成功，结果: {len(result)}行")
            results.append(True)
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            results.append(False)
    
    return all(results)


def test_mixed_datetime_keys():
    """测试混合日期时间键"""
    print("\n=== 测试混合日期时间键 ===")
    
    # 创建包含字符串和时间戳的混合键
    df1 = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'price': [150.0, 135.0, 140.0]
    })
    
    df2 = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'TSLA'],
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'volume': [50000000, 25000000, 40000000]
    })
    
    print("测试混合键 [symbol, date]:")
    print("数据1:", df1.to_dict('records'))
    print("数据2:", df2.to_dict('records'))
    
    try:
        result = rp.fast_inner_join_df(df1, df2, on=['symbol', 'date'])
        print(f"✅ 混合键合并成功: {len(result)}行")
        print(result)
        return True
        
    except Exception as e:
        print(f"❌ 混合键合并失败: {e}")
        return False


def validate_pandas_compatibility():
    """验证与pandas的兼容性"""
    print("\n=== 验证pandas兼容性 ===")
    
    # 创建测试数据
    dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
    
    df1 = pd.DataFrame({
        'date': dates,
        'value1': [100, 200, 300]
    })
    
    df2 = pd.DataFrame({
        'date': dates[:2],  # 只取前两个
        'value2': [10, 20]
    })
    
    try:
        # rust_pyfunc结果
        rust_result = rp.fast_inner_join_df(df1, df2, on='date')
        
        # pandas结果
        pandas_result = pd.merge(df1, df2, on='date', how='inner')
        
        print(f"rust_pyfunc结果: {len(rust_result)}行")
        print(f"pandas结果: {len(pandas_result)}行")
        
        # 验证行数一致性
        if len(rust_result) == len(pandas_result):
            print("✅ 结果行数一致")
            return True
        else:
            print("❌ 结果行数不一致")
            return False
            
    except Exception as e:
        print(f"❌ 兼容性测试失败: {e}")
        return False


def main():
    """运行所有验证测试"""
    print("开始验证Timestamp类型错误修复\n")
    
    tests = [
        ("原始错误重现", reproduce_original_error),
        ("各种日期时间类型", test_various_datetime_types),
        ("混合日期时间键", test_mixed_datetime_keys),
        ("pandas兼容性", validate_pandas_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            print(f"\n{test_name}: {'✅ 通过' if result else '❌ 失败'}")
        except Exception as e:
            print(f"\n{test_name}: ❌ 异常 - {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("修复验证结果汇总:")
    print("="*60)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ 通过" if results[i] else "❌ 失败"
        print(f"{test_name}: {status}")
    
    overall_success = all(results)
    print(f"\n总体结果: {'✅ 所有测试通过，修复成功！' if overall_success else '❌ 部分测试失败'}")
    
    if overall_success:
        print("\n🎉 Timestamp类型错误已完全修复！")
        print("💡 现在支持的日期时间类型：")
        print("   • pandas.Timestamp")
        print("   • datetime.date")
        print("   • datetime.datetime")
        print("   • numpy.datetime64")
        print("   • 以及它们的混合组合键")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)