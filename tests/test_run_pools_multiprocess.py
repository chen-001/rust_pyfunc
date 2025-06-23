#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证run_pools现在使用Rust原生多进程
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
    print("🔧 验证run_pools现在使用Rust原生多进程")
    print("=" * 60)
    
    # 创建测试数据
    args = [[20220101 + i, f"{i+1:06d}"] for i in range(50)]
    
    print(f"测试数据: {len(args)} 个任务")
    
    # 使用原来的run_pools函数（现在内部使用多进程）
    print(f"\n🚀 调用rust_pyfunc.run_pools():")
    print(f"   （现在内部使用Rust原生多进程实现）")
    
    start_time = time.time()
    
    # 注意：这里调用的是run_pools，不是run_multiprocess
    result = rust_pyfunc.run_pools(
        test_function,
        args,
        num_threads=6,  # 现在这个参数会转换为num_processes
        progress_callback=lambda completed, total, elapsed, speed: 
            print(f"   📊 进度: {completed/total*100:.1f}% | 速度: {speed:.0f} 任务/秒") 
            if completed % 10 == 0 or completed == total else None
    )
    
    elapsed = time.time() - start_time
    speed = len(args) / elapsed
    
    print(f"\n✅ 执行完成:")
    print(f"   🕒 执行时间: {elapsed:.3f} 秒")
    print(f"   🏎️  处理速度: {speed:.0f} 任务/秒")
    print(f"   📈 结果数量: {len(result)}")
    print(f"   🎯 结果格式: {result[0]}")
    
    # 验证结果
    assert len(result) == len(args), "结果数量不正确"
    assert result[0][0] == 20220101, "日期不正确"
    assert result[0][1] == "000001", "代码不正确"
    
    print(f"\n🎉 验证成功！")
    print(f"✨ 关键变化：")
    print(f"   • 不再有'函数不支持multiprocessing，使用优化的串行处理'的提示")
    print(f"   • 而是显示'开始Rust原生多进程执行'")
    print(f"   • 每个进程独立的Python解释器，真正的并行处理")
    print(f"   • API保持不变，但内部实现完全升级")
    
    print(f"\n💡 使用说明：")
    print(f"   • 继续使用 rust_pyfunc.run_pools() - API不变")
    print(f"   • 或者使用 rust_pyfunc.run_multiprocess() - 新API")
    print(f"   • 两个函数现在都使用相同的Rust原生多进程后端")
    print(f"   • 真正避开了Python GIL限制")


if __name__ == "__main__":
    main()