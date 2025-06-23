#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Web服务器功能
"""

import sys
import tempfile
import time
import threading
import requests
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc
from rust_pyfunc.web_manager import start_web_manager, BackupWebManager


def test_web_server_basic():
    """测试Web服务器基本功能"""
    print("=== 测试Web服务器基本功能 ===")
    
    # 使用不同的端口避免冲突
    port = 5001
    
    try:
        # 创建临时目录用于备份文件
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"临时目录: {temp_dir}")
            
            # 创建一些测试备份文件
            test_data = [
                (20220101, "000001"),
                (20220101, "000002"),
                (20220102, "000001"),
            ]
            
            # 创建简单的测试函数
            def simple_func(date, code):
                return [float(date % 1000), float(len(code)), 1.0]
            
            # 生成测试备份
            backup_file = f"{temp_dir}/test_backup.json"
            result = rust_pyfunc.run_pools(
                simple_func,
                [[date, code] for date, code in test_data],  # 转换为列表格式
                backup_file=backup_file,
                storage_format="json",
                num_threads=2
            )
            
            print(f"生成了测试备份文件，结果数量: {len(result)}")
            
            # 创建Web管理器
            manager = BackupWebManager(
                backup_directory=temp_dir,
                host="127.0.0.1",
                port=port
            )
            
            # 在单独线程中启动服务器
            def run_server():
                try:
                    manager.run(debug=False)
                except Exception as e:
                    print(f"服务器启动失败: {e}")
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # 等待服务器启动
            time.sleep(2)
            
            # 测试主页访问
            try:
                response = requests.get(f"http://127.0.0.1:{port}", timeout=5)
                print(f"主页访问状态码: {response.status_code}")
                
                if response.status_code == 200:
                    print("✓ Web服务器运行正常")
                    print(f"页面标题: {'备份数据管理界面' if '备份数据管理界面' in response.text else '未找到预期标题'}")
                else:
                    print(f"❌ Web服务器返回错误状态码: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ 无法连接到Web服务器: {e}")
                
            # 测试API接口
            try:
                api_response = requests.get(f"http://127.0.0.1:{port}/api/backups", timeout=5)
                print(f"API访问状态码: {api_response.status_code}")
                
                if api_response.status_code == 200:
                    backups = api_response.json()
                    print(f"✓ API正常，找到 {len(backups)} 个备份文件")
                else:
                    print(f"❌ API返回错误状态码: {api_response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ API请求失败: {e}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_port_conflict():
    """测试端口冲突情况"""
    print("\n=== 测试端口冲突处理 ===")
    
    # 尝试使用已被占用的端口5000
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            manager = BackupWebManager(
                backup_directory=temp_dir,
                host="127.0.0.1",
                port=5000  # 已被占用的端口
            )
            
            # 短时间运行测试
            def run_server():
                try:
                    manager.run(debug=False)
                except Exception as e:
                    print(f"端口冲突错误（预期的）: {e}")
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            time.sleep(1)
            print("端口冲突测试完成")
            
        except Exception as e:
            print(f"端口冲突处理: {e}")


if __name__ == "__main__":
    print("开始测试Web服务器功能")
    print("=" * 50)
    
    try:
        test_web_server_basic()
        test_port_conflict()
        
        print("\n" + "=" * 50)
        print("🎉 Web服务器测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)