#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Web服务器自动端口选择功能
"""

import sys
import tempfile
import time
import threading
import requests
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

from rust_pyfunc.web_manager import start_web_manager, check_port_available, find_available_port


def test_port_functions():
    """测试端口检查函数"""
    print("=== 测试端口检查函数 ===")
    
    # 测试已被占用的端口5000
    port_5000_available = check_port_available("127.0.0.1", 5000)
    print(f"端口5000是否可用: {port_5000_available}")
    
    # 寻找可用端口
    try:
        available_port = find_available_port("127.0.0.1", 5000)
        print(f"找到可用端口: {available_port}")
        
        # 验证找到的端口确实可用
        is_available = check_port_available("127.0.0.1", available_port)
        print(f"端口{available_port}确实可用: {is_available}")
        
    except RuntimeError as e:
        print(f"寻找端口失败: {e}")


def test_auto_port_web_server():
    """测试自动端口选择的Web服务器"""
    print("\n=== 测试自动端口选择 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"临时目录: {temp_dir}")
        
        # 启动Web服务器（会自动选择可用端口）
        server_started = threading.Event()
        actual_port = None
        
        def run_server():
            nonlocal actual_port
            try:
                # 由于端口5000被占用，应该自动选择其他端口
                start_web_manager(
                    backup_directory=temp_dir,
                    port=5000,  # 尝试使用被占用的端口
                    debug=False,
                    auto_port=True
                )
            except Exception as e:
                print(f"服务器启动失败: {e}")
            finally:
                server_started.set()
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # 等待服务器启动（或失败）
        server_started.wait(timeout=5)
        time.sleep(1)
        
        # 尝试在多个可能的端口上测试
        for test_port in range(5001, 5010):
            try:
                response = requests.get(f"http://127.0.0.1:{test_port}", timeout=2)
                if response.status_code == 200:
                    print(f"✓ Web服务器在端口{test_port}上运行正常")
                    actual_port = test_port
                    break
            except requests.exceptions.RequestException:
                continue
        
        if actual_port is None:
            print("❌ 未能找到运行中的Web服务器")
        else:
            print(f"✓ 自动端口选择功能工作正常，实际端口: {actual_port}")


if __name__ == "__main__":
    print("开始测试Web服务器自动端口选择功能")
    print("=" * 50)
    
    try:
        test_port_functions()
        test_auto_port_web_server()
        
        print("\n" + "=" * 50)
        print("🎉 自动端口选择测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)