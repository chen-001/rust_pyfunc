"""Web管理器模块类型声明"""
from typing import Optional, Tuple, List, Dict, Any
import flask

def check_port_available(host: str, port: int) -> bool:
    """检查端口是否可用。
    
    参数说明：
    ----------
    host : str
        主机地址
    port : int
        端口号
        
    返回值：
    -------
    bool
        端口是否可用
    """
    ...

def find_available_port(host: str, start_port: int, max_attempts: int = 10) -> Optional[int]:
    """寻找可用端口。
    
    参数说明：
    ----------
    host : str
        主机地址
    start_port : int
        起始端口号
    max_attempts : int, 默认10
        最大尝试次数
        
    返回值：
    -------
    Optional[int]
        可用端口号，如果没找到则返回None
    """
    ...

def start_web_manager(
    backup_directory: str = "./",
    host: str = "127.0.0.1", 
    port: int = 5000,
    debug: bool = False,
    auto_port: bool = True
) -> None:
    """启动备份数据Web管理界面。
    
    参数说明：
    ----------
    backup_directory : str, 默认"./"
        备份文件目录
    host : str, 默认"127.0.0.1"
        服务器主机地址
    port : int, 默认5000
        服务器端口
    debug : bool, 默认False
        是否启用调试模式
    auto_port : bool, 默认True
        是否自动寻找可用端口
    """
    ...

class BackupWebManager:
    """备份数据Web管理器。
    
    提供Web界面来管理和查看备份文件。
    """
    
    def __init__(
        self,
        backup_directory: str = "./",
        host: str = "127.0.0.1",
        port: int = 5000
    ) -> None:
        """初始化Web管理器。
        
        参数说明：
        ----------
        backup_directory : str, 默认"./"
            备份文件目录
        host : str, 默认"127.0.0.1"
            服务器主机地址
        port : int, 默认5000
            服务器端口
        """
        ...
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """列出所有备份文件。
        
        返回值：
        -------
        List[Dict[str, Any]]
            备份文件信息列表
        """
        ...
    
    def run(self, debug: bool = False) -> None:
        """启动Web服务器。
        
        参数说明：
        ----------
        debug : bool, 默认False
            是否启用调试模式
        """
        ...