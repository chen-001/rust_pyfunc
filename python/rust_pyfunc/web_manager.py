#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
备份数据Web管理界面
==================

提供Web界面来管理和查询备份数据。

功能:
- 列出所有备份文件
- 查看备份文件信息
- 查询备份数据
- 删除备份文件
"""

import os
import glob
import json
import tempfile
from datetime import datetime
from typing import List, Optional, Dict, Any

try:
    from flask import Flask, render_template_string, request, jsonify, redirect, url_for
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

import rust_pyfunc

class BackupWebManager:
    """备份数据Web管理器"""
    
    def __init__(self, backup_directory: str = "./", host: str = "127.0.0.1", port: int = 5000):
        """
        初始化Web管理器
        
        参数:
            backup_directory: 备份文件目录
            host: Web服务器主机
            port: Web服务器端口
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask未安装，请运行: pip install flask")
            
        self.backup_directory = os.path.abspath(backup_directory)
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/')
        def index():
            """主页 - 列出所有备份文件"""
            backups = self.list_backups()
            return render_template_string(MAIN_TEMPLATE, backups=backups)
        
        @self.app.route('/api/backups')
        def api_backups():
            """API - 获取备份列表"""
            backups = self.list_backups()
            return jsonify(backups)
        
        @self.app.route('/api/query', methods=['POST'])
        def api_query():
            """API - 查询备份数据"""
            data = request.get_json()
            backup_file = data.get('backup_file')
            storage_format = data.get('storage_format', 'json')
            date_range = data.get('date_range')
            codes = data.get('codes')
            
            try:
                # 查询数据
                results = rust_pyfunc.query_backup(
                    backup_file=backup_file,
                    date_range=tuple(date_range) if date_range else None,
                    codes=codes,
                    storage_format=storage_format
                )
                
                # 转换NDArray为列表
                if hasattr(results, 'tolist'):
                    results_list = results.tolist()
                else:
                    results_list = results
                
                return jsonify({
                    'success': True,
                    'data': results_list,
                    'count': len(results_list)
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/api/delete', methods=['POST'])
        def api_delete():
            """API - 删除备份文件"""
            data = request.get_json()
            backup_file = data.get('backup_file')
            storage_format = data.get('storage_format', 'json')
            
            try:
                rust_pyfunc.delete_backup(backup_file, storage_format)
                return jsonify({'success': True})
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """列出所有备份文件"""
        backups = []
        
        # 搜索不同格式的备份文件
        patterns = ['*.json', '*.bin', '*.parquet']
        
        for pattern in patterns:
            files = glob.glob(os.path.join(self.backup_directory, pattern))
            
            for file_path in files:
                # 确定存储格式
                if file_path.endswith('.json'):
                    storage_format = 'json'
                elif file_path.endswith('.bin'):
                    storage_format = 'binary'
                else:
                    storage_format = 'memory_map'
                
                try:
                    # 获取文件信息
                    size, modified_time = rust_pyfunc.get_backup_info(file_path, storage_format)
                    
                    backups.append({
                        'file_path': file_path,
                        'filename': os.path.basename(file_path),
                        'storage_format': storage_format,
                        'size': size,
                        'size_mb': round(size / 1024 / 1024, 2),
                        'modified_time': modified_time,
                        'exists': True
                    })
                except Exception as e:
                    # 文件可能损坏或不可读
                    backups.append({
                        'file_path': file_path,
                        'filename': os.path.basename(file_path),
                        'storage_format': storage_format,
                        'size': 0,
                        'size_mb': 0,
                        'modified_time': '未知',
                        'exists': False,
                        'error': str(e)
                    })
        
        return sorted(backups, key=lambda x: x['filename'])
    
    def run(self, debug: bool = False):
        """启动Web服务器"""
        print(f"🌐 备份管理Web界面启动中...")
        print(f"📂 备份目录: {self.backup_directory}")
        
        # 尝试启动服务器，如果端口被占用则自动寻找可用端口
        max_attempts = 10
        original_port = self.port
        
        for attempt in range(max_attempts):
            try:
                print(f"🔗 尝试启动在: http://{self.host}:{self.port}")
                self.app.run(host=self.host, port=self.port, debug=debug)
                break
            except OSError as e:
                if "Address already in use" in str(e) or "端口已被占用" in str(e):
                    self.port += 1
                    if attempt < max_attempts - 1:
                        print(f"⚠️  端口 {self.port - 1} 已被占用，尝试端口 {self.port}")
                    else:
                        print(f"❌ 无法找到可用端口（尝试了 {original_port} 到 {self.port}）")
                        raise
                else:
                    raise
        
        print(f"💡 在浏览器中打开 http://{self.host}:{self.port} 来管理备份文件")

# HTML模板
MAIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>备份数据管理界面</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        .backup-list {
            margin-bottom: 30px;
        }
        .backup-item {
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 15px;
            padding: 15px;
            background-color: #fafafa;
        }
        .backup-item.error {
            border-color: #dc3545;
            background-color: #f8d7da;
        }
        .backup-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .backup-filename {
            font-weight: bold;
            font-size: 16px;
            color: #007bff;
        }
        .backup-format {
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }
        .format-json { background-color: #28a745; }
        .format-binary { background-color: #6610f2; }
        .format-memory_map { background-color: #fd7e14; }
        .backup-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            font-size: 14px;
            color: #666;
        }
        .backup-actions {
            margin-top: 10px;
        }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
            font-size: 14px;
            text-decoration: none;
            display: inline-block;
        }
        .btn-primary {
            background-color: #007bff;
            color: white;
        }
        .btn-danger {
            background-color: #dc3545;
            color: white;
        }
        .btn:hover {
            opacity: 0.8;
        }
        .query-section {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            margin-top: 30px;
            background-color: #f8f9fa;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .form-control {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .results-table th,
        .results-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .results-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .results-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .error-message {
            color: #dc3545;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
        }
        .success-message {
            color: #155724;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🗂️ 备份数据管理界面</h1>
        
        <div class="backup-list">
            <h2>📁 备份文件列表</h2>
            {% if backups %}
                {% for backup in backups %}
                <div class="backup-item {% if not backup.exists %}error{% endif %}">
                    <div class="backup-header">
                        <div class="backup-filename">{{ backup.filename }}</div>
                        <div class="backup-format format-{{ backup.storage_format }}">{{ backup.storage_format.upper() }}</div>
                    </div>
                    <div class="backup-info">
                        <div><strong>文件路径:</strong> {{ backup.file_path }}</div>
                        <div><strong>文件大小:</strong> {{ backup.size_mb }} MB</div>
                        <div><strong>修改时间:</strong> {{ backup.modified_time }}</div>
                        {% if backup.error %}
                        <div><strong>错误:</strong> {{ backup.error }}</div>
                        {% endif %}
                    </div>
                    {% if backup.exists %}
                    <div class="backup-actions">
                        <button class="btn btn-primary" onclick="queryBackup('{{ backup.file_path }}', '{{ backup.storage_format }}')">查询数据</button>
                        <button class="btn btn-danger" onclick="deleteBackup('{{ backup.file_path }}', '{{ backup.storage_format }}')">删除文件</button>
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            {% else %}
                <p style="text-align: center; color: #666; padding: 40px;">
                    📝 暂无备份文件<br>
                    使用 <code>rust_pyfunc.run_pools()</code> 创建备份文件
                </p>
            {% endif %}
        </div>
        
        <div class="query-section">
            <h2>🔍 数据查询</h2>
            <form onsubmit="return false;">
                <div class="form-group">
                    <label for="query-file">备份文件:</label>
                    <select id="query-file" class="form-control">
                        <option value="">请选择备份文件</option>
                        {% for backup in backups %}
                        {% if backup.exists %}
                        <option value="{{ backup.file_path }}" data-format="{{ backup.storage_format }}">{{ backup.filename }} ({{ backup.storage_format }})</option>
                        {% endif %}
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label for="query-date-start">开始日期 (YYYYMMDD):</label>
                    <input type="text" id="query-date-start" class="form-control" placeholder="例如: 20220101">
                </div>
                <div class="form-group">
                    <label for="query-date-end">结束日期 (YYYYMMDD):</label>
                    <input type="text" id="query-date-end" class="form-control" placeholder="例如: 20220131">
                </div>
                <div class="form-group">
                    <label for="query-codes">股票代码 (用逗号分隔):</label>
                    <input type="text" id="query-codes" class="form-control" placeholder="例如: 000001,000002,600000">
                </div>
                <button type="button" class="btn btn-primary" onclick="executeQuery()">查询数据</button>
            </form>
            
            <div id="query-results"></div>
        </div>
    </div>
    
    <script>
        function queryBackup(filePath, storageFormat) {
            document.getElementById('query-file').value = filePath;
            document.getElementById('query-file').scrollIntoView();
        }
        
        async function deleteBackup(filePath, storageFormat) {
            if (!confirm('确定要删除备份文件 "' + filePath + '" 吗？\\n此操作不可撤销！')) {
                return;
            }
            
            try {
                const response = await fetch('/api/delete', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        backup_file: filePath,
                        storage_format: storageFormat
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert('备份文件删除成功！');
                    location.reload();
                } else {
                    alert('删除失败: ' + result.error);
                }
            } catch (error) {
                alert('删除请求失败: ' + error.message);
            }
        }
        
        async function executeQuery() {
            const fileSelect = document.getElementById('query-file');
            const filePath = fileSelect.value;
            const storageFormat = fileSelect.selectedOptions[0]?.dataset.format || 'json';
            const dateStart = document.getElementById('query-date-start').value;
            const dateEnd = document.getElementById('query-date-end').value;
            const codes = document.getElementById('query-codes').value;
            
            if (!filePath) {
                alert('请选择要查询的备份文件');
                return;
            }
            
            const queryData = {
                backup_file: filePath,
                storage_format: storageFormat
            };
            
            if (dateStart && dateEnd) {
                queryData.date_range = [parseInt(dateStart), parseInt(dateEnd)];
            }
            
            if (codes) {
                queryData.codes = codes.split(',').map(s => s.trim()).filter(s => s);
            }
            
            const resultsDiv = document.getElementById('query-results');
            resultsDiv.innerHTML = '<div class="loading">🔄 查询中...</div>';
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(queryData)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayResults(result.data, result.count);
                } else {
                    resultsDiv.innerHTML = '<div class="error-message">查询失败: ' + result.error + '</div>';
                }
            } catch (error) {
                resultsDiv.innerHTML = '<div class="error-message">查询请求失败: ' + error.message + '</div>';
            }
        }
        
        function displayResults(data, count) {
            const resultsDiv = document.getElementById('query-results');
            
            if (count === 0) {
                resultsDiv.innerHTML = '<div class="success-message">查询完成，未找到匹配的数据</div>';
                return;
            }
            
            let html = '<div class="success-message">查询完成，找到 ' + count + ' 条记录</div>';
            
            if (data.length > 0) {
                html += '<table class="results-table">';
                html += '<thead><tr>';
                
                // 表头
                const firstRow = data[0];
                const headers = ['日期', '股票代码'];
                if (firstRow.length > 2) {
                    if (firstRow.length > 3) {
                        headers.push('时间戳');
                        for (let i = 3; i < firstRow.length; i++) {
                            headers.push('因子' + (i - 2));
                        }
                    } else {
                        for (let i = 2; i < firstRow.length; i++) {
                            headers.push('因子' + (i - 1));
                        }
                    }
                }
                
                headers.forEach(header => {
                    html += '<th>' + header + '</th>';
                });
                html += '</tr></thead><tbody>';
                
                // 数据行 (只显示前100行)
                const maxRows = Math.min(data.length, 100);
                for (let i = 0; i < maxRows; i++) {
                    html += '<tr>';
                    data[i].forEach(cell => {
                        html += '<td>' + cell + '</td>';
                    });
                    html += '</tr>';
                }
                
                html += '</tbody></table>';
                
                if (data.length > 100) {
                    html += '<p style="text-align: center; color: #666; margin-top: 10px;">只显示前100行，共' + data.length + '行数据</p>';
                }
            }
            
            resultsDiv.innerHTML = html;
        }
    </script>
</body>
</html>
"""

def check_port_available(host: str, port: int) -> bool:
    """检查端口是否可用"""
    import socket
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result != 0  # 0表示连接成功（端口被占用）
    except Exception:
        return False


def find_available_port(host: str, start_port: int, max_attempts: int = 10) -> int:
    """寻找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        if check_port_available(host, port):
            return port
    raise RuntimeError(f"无法在 {start_port} 到 {start_port + max_attempts - 1} 范围内找到可用端口")


def start_web_manager(backup_directory: str = "./", host: str = "127.0.0.1", port: int = 5000, debug: bool = False, auto_port: bool = True):
    """
    启动备份数据Web管理界面
    
    参数:
        backup_directory: 备份文件目录，默认当前目录
        host: Web服务器主机，默认本地主机
        port: Web服务器端口，默认5000
        debug: 是否开启调试模式
        auto_port: 如果指定端口被占用，是否自动寻找可用端口
    
    示例:
        >>> import rust_pyfunc.web_manager as web
        >>> web.start_web_manager()  # 启动在 http://127.0.0.1:5000
        >>> web.start_web_manager(port=8080)  # 启动在指定端口
    """
    if auto_port and not check_port_available(host, port):
        try:
            available_port = find_available_port(host, port)
            print(f"⚠️  端口 {port} 已被占用，自动选择端口 {available_port}")
            port = available_port
        except RuntimeError as e:
            print(f"❌ {e}")
            return
    
    manager = BackupWebManager(backup_directory, host, port)
    manager.run(debug=debug)

if __name__ == "__main__":
    start_web_manager()