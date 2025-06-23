#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的多进程测试
"""

import sys
import time
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc


def test_function(date, code):
    """测试函数"""
    # 模拟CPU密集型计算
    result = 0
    for i in range(1000):
        result += hash(f"{date}_{code}_{i}") % 100
    
    return [
        float(date % 10000),
        float(len(code)),
        float(result % 1000)
    ]


def main():
    print("🚀 Rust原生多进程测试")
    print("=" * 50)
    
    # 创建测试数据
    args = [[20220101 + i, f"{i+1:06d}"] for i in range(20)]
    
    print(f"测试数据: {len(args)} 个任务")
    print(f"每个任务执行1000次hash计算")
    
    # 测试多进程
    print(f"\n🔧 使用Rust原生多进程 (4进程):")
    start_time = time.time()
    
    result = rust_pyfunc.run_multiprocess(
        test_function,
        args,
        num_processes=4,
        progress_callback=lambda completed, total, elapsed, speed: 
            print(f"   进度: {completed/total*100:.1f}% | 速度: {speed:.0f} 任务/秒") 
            if completed % 5 == 0 or completed == total else None
    )
    
    elapsed = time.time() - start_time
    speed = len(args) / elapsed
    
    print(f"\n✅ 多进程执行完成:")
    print(f"   执行时间: {elapsed:.3f} 秒")
    print(f"   处理速度: {speed:.0f} 任务/秒")
    print(f"   结果数量: {len(result)}")
    print(f"   结果示例: {result[0]}")
    
    # 验证结果
    assert len(result) == len(args), "结果数量不正确"
    assert result[0][0] == 20220101, "日期不正确"
    assert result[0][1] == "000001", "代码不正确"
    
    print(f"\n🎉 测试成功！")
    print(f"✨ 真正的多进程并行：每个进程独立的Python解释器，无GIL限制")
    print(f"🚀 Rust系统级控制：直接管理进程生命周期，高性能执行")


if __name__ == "__main__":
    main()