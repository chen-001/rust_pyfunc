#!/usr/bin/env python3
"""
独立的进度监控进程
通过读取备份文件来推断计算进度，不占用主进程资源
使用WebTqdmforRust在5101端口显示进度和最新数据
"""
import sys
import os
import time
import argparse
import struct
from pathlib import Path
from typing import List, Dict, Any, Optional
import signal
import datetime
import uuid
import numpy as np
import requests

# 添加design_whatever到路径
sys.path.append('/home/chenzongwei/design_whatever')

# 导入rust_pyfunc来读取备份数据
try:
    import rust_pyfunc
except ImportError:
    print("警告: 无法导入rust_pyfunc，将无法读取真实备份数据")
    rust_pyfunc = None

class WebTqdmforRust:
    """
    增强版进度条，用于监控Rust多进程的分块进度。
    - 网页实时监控
    - 错误状态上报
    - 接收分块进度和数据
    """

    def __init__(self, total, name=None, server_url='http://localhost:5101', fac_names=None):
        self.total = total  # 总块数
        self.n = 0          # 已完成块数
        self.task_name = name
        self.server_url = server_url
        self.task_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
        self.table_data = []
        self.fac_names = fac_names or []  # 因子名称列表
        self._error_occurred = False
        print(f"WebTqdm已启动, 任务: {self.task_name or 'Untitled'}, ID: {self.task_id}, 总块数: {self.total}")

    def update(self, progress_info):
        """由Rust回调，接收分块进度信息 [completed_chunks, total_chunks, remaining, elapsed]"""
        if not progress_info or len(progress_info) < 4:
            return

        self.n, self.total, remaining, elapsed = progress_info
        self._post_update(status='running')

    def _update_table_data(self, chunk_results):
        """由Rust回调，接收一个数据块的计算结果"""
        # 处理数据：转换时间戳、展开因子、只保留最后20行
        processed_data = []
        
        for row in chunk_results:
            processed_row = {}
            
            # 复制基本字段
            processed_row['date'] = row.get('date')
            processed_row['code'] = row.get('code')
            
            # 转换时间戳为可读格式
            timestamp = row.get('timestamp', 0)
            if timestamp:
                dt = datetime.datetime.fromtimestamp(timestamp)
                processed_row['timestamp'] = dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                processed_row['timestamp'] = '1970-01-01 00:00:00'
            
            # 处理因子数据
            facs = row.get('facs', [])
            if self.fac_names and len(self.fac_names) >= len(facs):
                # 如果提供了因子名称，则展开为单独的列
                for i, fac_value in enumerate(facs):
                    fac_name = self.fac_names[i]
                    processed_row[fac_name] = fac_value
            else:
                # 否则保持原来的facs格式
                processed_row['facs'] = facs
                
            processed_data.append(processed_row)
        
        # 只保留最后20行
        self.table_data = processed_data[-20:] if len(processed_data) > 20 else processed_data
        
        # 在数据更新后也上报一次，确保网页能看到最新数据
        self._post_update(status='running')

    def set_error(self, error_info="Unknown error"):
        """设置错误状态"""
        self._error_occurred = True
        self._post_update(status='error', error=error_info)

    def finish(self):
        """标记任务完成"""
        if not self._error_occurred:
            self.n = self.total # 确保进度条达到100%
            self._post_update(status='completed', is_final=True)

    def _post_update(self, status='running', error=None, is_final=False):
        """上报数据到服务器"""
        elapsed = time.time() - self.start_time
        
        if self.n > 0 and self.total and self.total > self.n:
            remaining = (elapsed / self.n) * (self.total - self.n)
        else:
            remaining = 0
        
        # 处理表格数据中的特殊值
        def clean_value(v):
            if isinstance(v, float):
                if np.isinf(v) or np.isnan(v):
                    return None
                return float(v)
            return v

        cleaned_table_data = []
        if isinstance(self.table_data, list):
            for row in self.table_data:
                cleaned_row = {k: clean_value(v) for k, v in row.items()}
                cleaned_table_data.append(cleaned_row)
        
        try:
            requests.post(
                f"{self.server_url}/update",
                json={
                    "task_id": self.task_id,
                    "name": self.task_name or f"Task-{self.task_id}",
                    "progress": float(self.n / self.total * 100 if self.total else 0),
                    "elapsed": float(elapsed),
                    "remaining": float(remaining),
                    "table_data": cleaned_table_data,
                    "status": status,
                    "error": error
                },
                timeout=1
            )
        except Exception as e:
            # 在生产环境中，可以考虑更优雅的日志记录方式
            # print(f"更新进度失败: {str(e)}")
            pass

class BackupFileMonitor:
    """备份文件监控器，通过分析.bin文件推断进度"""
    
    def __init__(self, backup_file: str, total_tasks: int, task_name: str = None, factor_names: List[str] = None):
        self.backup_file = backup_file
        self.total_tasks = total_tasks
        self.task_name = task_name or f"任务监控-{os.path.basename(backup_file)}"
        self.factor_names = factor_names or []
        
        # 创建WebTqdmforRust实例
        self.web_tqdm = WebTqdmforRust(
            total=total_tasks,
            name=self.task_name,
            fac_names=self.factor_names
        )
        
        self.last_size = 0
        self.last_count = 0
        self.start_time = time.time()
        self.running = True
        
        print(f"开始监控备份文件: {backup_file}")
        print(f"总任务数: {total_tasks}")
        print(f"WebTqdm ID: {self.web_tqdm.task_id}")
        print("请在浏览器中打开 http://localhost:5101 查看进度")
        
    def parse_binary_backup(self) -> List[Dict[str, Any]]:
        """使用rust_pyfunc读取真实的备份数据"""
        if not os.path.exists(self.backup_file):
            return []
            
        if rust_pyfunc is None:
            print("rust_pyfunc未导入，无法读取备份数据")
            return []
            
        try:
            # 使用rust_pyfunc的query_backup函数读取最新数据
            # 不指定日期和股票范围，读取所有数据
            backup_array = rust_pyfunc.query_backup(
                backup_file=self.backup_file,
                date_range=None,
                codes=None,
                storage_format="binary"
            )
            
            if backup_array.size == 0:
                return []
            
            # 转换NDArray为Python列表格式
            results = []
            num_rows, num_cols = backup_array.shape
            
            # backup_array格式: [date, code, timestamp, fac1, fac2, ...]
            for i in range(num_rows):
                row = backup_array[i]
                result = {
                    'date': int(row[0]),
                    'code': str(row[1]),
                    'timestamp': int(row[2]),
                    'facs': [float(row[j]) for j in range(3, num_cols)]  # 从第3列开始是因子数据
                }
                results.append(result)
            
            # 只返回最新的20条记录
            recent_results = results[-20:] if len(results) > 20 else results
            print(f"成功读取备份数据: {len(recent_results)} 条记录")
            return recent_results
            
        except Exception as e:
            print(f"读取备份数据失败: {e}")
            # 如果读取失败，尝试通过文件大小估算一些基本信息
            try:
                file_size = os.path.getsize(self.backup_file)
                if file_size > 0:
                    print(f"备份文件存在，大小: {file_size} 字节，但无法解析内容")
            except:
                pass
            return []
    
            
    def estimate_progress_from_backup(self) -> int:
        """通过读取备份文件中的实际记录数来获取准确进度"""
        if not os.path.exists(self.backup_file):
            return 0
            
        if rust_pyfunc is None:
            # 如果无法导入rust_pyfunc，回退到文件大小估算
            return self.estimate_progress_from_file_size()
            
        try:
            # 使用rust_pyfunc直接读取备份记录总数
            backup_array = rust_pyfunc.query_backup(
                backup_file=self.backup_file,
                date_range=None,
                codes=None,
                storage_format="binary"
            )
            
            if backup_array.size == 0:
                return 0
                
            # 返回实际记录数
            actual_count = backup_array.shape[0]
            self.last_count = min(actual_count, self.total_tasks)
            return self.last_count
            
        except Exception as e:
            print(f"读取备份记录数失败，回退到文件大小估算: {e}")
            return self.estimate_progress_from_file_size()
    
    def estimate_progress_from_file_size(self) -> int:
        """通过文件大小估算完成的任务数（备用方法）"""
        if not os.path.exists(self.backup_file):
            return 0
            
        try:
            current_size = os.path.getsize(self.backup_file)
            
            # 如果文件大小没有变化，返回之前的计数
            if current_size == self.last_size:
                return self.last_count
                
            # 粗略估算：假设每个任务平均产生100字节的备份数据
            estimated_count = min(current_size // 100, self.total_tasks)
            
            self.last_size = current_size
            self.last_count = estimated_count
            
            return estimated_count
            
        except Exception as e:
            print(f"估算进度失败: {e}")
            return self.last_count
    
    def update_progress(self):
        """更新进度信息"""
        try:
            # 获取当前完成的任务数（优先使用准确的备份记录数）
            completed = self.estimate_progress_from_backup()
            elapsed = time.time() - self.start_time
            
            # 计算剩余时间
            if completed > 0 and completed < self.total_tasks:
                remaining = (elapsed / completed) * (self.total_tasks - completed)
            else:
                remaining = 0.0
                
            # 更新WebTqdm进度
            progress_info = [completed, self.total_tasks, remaining, elapsed]
            self.web_tqdm.update(progress_info)
            
            print(f"进度: {completed}/{self.total_tasks} ({completed/self.total_tasks*100:.1f}%), "
                  f"已用时: {elapsed:.1f}s, 预计剩余: {remaining:.1f}s")
                  
        except Exception as e:
            print(f"更新进度失败: {e}")
            self.web_tqdm.set_error(str(e))
    
    def update_table_data(self):
        """更新表格数据（低频率）"""
        try:
            # 解析最新的备份数据
            recent_results = self.parse_binary_backup()
            
            if recent_results:
                print(f"更新表格数据: {len(recent_results)} 条记录")
                self.web_tqdm._update_table_data(recent_results)
            
        except Exception as e:
            print(f"更新表格数据失败: {e}")
    
    def run(self):
        """主监控循环"""
        update_count = 0
        
        try:
            while self.running:
                # 更新进度信息（每3秒一次）
                self.update_progress()
                
                # 更新表格数据（每30秒一次）
                update_count += 1
                if update_count % 10 == 0:  # 30秒 / 3秒 = 10次
                    self.update_table_data()
                
                # 检查是否完成
                if self.last_count >= self.total_tasks:
                    print("任务已完成!")
                    self.web_tqdm.finish()
                    break
                    
                time.sleep(3)  # 3秒更新一次进度
                
        except KeyboardInterrupt:
            print("收到中断信号，停止监控...")
        except Exception as e:
            print(f"监控过程中出错: {e}")
            self.web_tqdm.set_error(str(e))
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.running = False
        print("监控已停止")
    
    def signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"收到信号 {signum}，开始清理...")
        self.cleanup()
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='独立的进度监控进程')
    parser.add_argument('--backup-file', required=True, help='备份文件路径')
    parser.add_argument('--total-tasks', type=int, required=True, help='总任务数')
    parser.add_argument('--task-name', help='任务名称')
    parser.add_argument('--factor-names', nargs='*', help='因子名称列表')
    
    args = parser.parse_args()
    
    # 创建监控器
    monitor = BackupFileMonitor(
        backup_file=args.backup_file,
        total_tasks=args.total_tasks,
        task_name=args.task_name,
        factor_names=args.factor_names
    )
    
    # 设置信号处理
    signal.signal(signal.SIGINT, monitor.signal_handler)
    signal.signal(signal.SIGTERM, monitor.signal_handler)
    
    # 开始监控
    monitor.run()

if __name__ == "__main__":
    main()