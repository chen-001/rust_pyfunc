"""并行计算和备份管理函数类型声明"""
from typing import List, Optional, Tuple, Callable
import numpy as np
from numpy.typing import NDArray

def run_pools(
    func: Callable[[object, int, str], List[float]],
    args: List[Tuple[int, str]], 
    go_class: Optional[object] = None,
    num_threads: Optional[int] = None,
    backup_file: Optional[str] = None,
    backup_batch_size: int = 1000,
    storage_format: str = "binary",
    resume_from_backup: bool = False,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None
) -> NDArray[np.object_]:
    """高性能多进程并行执行函数，支持自动备份、进度监控和负载均衡。
    
    ⚡ Rust原生多进程架构（v0.19.2+）
    采用高效的多进程任务分配和结果收集机制，提供稳定的并行计算性能。
    
    参数说明：
    ----------
    func : Callable[[object, int, str], List[float]]
        要并行执行的Python函数，接受(go_class: object, date: int, code: str)参数，返回因子列表
    args : List[Tuple[int, str]]
        参数列表，每个元素是(date, code)元组
    go_class : Optional[object]
        Go类实例，将作为第一个参数传递给func函数，默认None
    num_threads : Optional[int]
        并行进程数，None表示自动检测CPU核心数
    backup_file : Optional[str]
        备份文件路径，None表示不备份
    backup_batch_size : int
        每批次备份的记录数，默认1000
    storage_format : str
        存储格式，支持"json", "binary", "memory_map"，默认"binary"
    resume_from_backup : bool
        是否从备份恢复，默认False
    progress_callback : Optional[Callable[[int, int, float, float], None]]
        进度回调函数，接受(completed, total, remaining_time, elapsed_time)参数
        
    返回值：
    -------
    NDArray[np.object_]
        结果数组，每行格式为[date, code, *facs]
        注意：如果启用了流式处理模式（结果为空），将从备份文件中读取数据
        
    🚀 多进程架构优势：
    ----------
    核心特性：
    - ✅ Rust原生多进程实现，高性能并行计算
    - ✅ 智能任务分配：自动将任务分配给可用进程
    - ✅ 流式处理支持：支持大数据量的流式备份和读取
    - ✅ 灵活的存储格式：支持JSON、二进制和内存映射格式
    - ✅ 实时进度监控：通过回调函数实时反馈处理进度
    
    性能特性：
    - 适用于大规模数据处理和因子计算任务
    - 支持备份恢复机制，确保数据安全
    - 自动管理进程生命周期，防止资源泄漏
    - 支持断点续传，从备份文件恢复未完成的任务
    - 推荐在高性能计算场景和批量数据处理中使用
        
    示例：
    -------
    >>> # 标准多进程处理模式
    >>> def my_analysis(go_class, date, code):
    ...     # 你的分析逻辑，可以使用go_class实例
    ...     return [1.0, 2.0, 3.0]  # 返回固定长度的因子列表
    >>> 
    >>> args = [(20220101, '000001'), (20220101, '000002')]
    >>> result = run_pools(
    ...     my_analysis, 
    ...     args,
    ...     go_class=my_go_instance,
    ...     backup_file="results.bin",
    ...     num_threads=8,
    ...     storage_format="binary"
    ... )
    >>> print(result)
    [[20220101 '000001' 1.0 2.0 3.0]
     [20220101 '000002' 1.0 2.0 3.0]]
     
    >>> # 大数据量处理示例（支持备份和恢复）
    >>> result = run_pools(
    ...     my_analysis,
    ...     large_args_list,  # 例如百万级任务
    ...     go_class=my_go_instance,
    ...     num_threads=16,
    ...     backup_file="large_results.bin",
    ...     backup_batch_size=5000,
    ...     resume_from_backup=True  # 支持断点续传
    ... )
    
    >>> # 带进度监控的处理示例
    >>> def progress_callback(completed, total, remaining_time, elapsed_time):
    ...     print(f"进度: {completed}/{total}, 剩余时间: {remaining_time:.2f}s")
    >>> 
    >>> result = run_pools(
    ...     my_analysis,
    ...     args,
    ...     progress_callback=progress_callback,
    ...     num_threads=8
    ... )
    """
    ...

def query_backup(
    backup_file: str,
    date_range: Optional[Tuple[int, int]] = None,
    codes: Optional[List[str]] = None,
    storage_format: str = "binary"
) -> NDArray[np.object_]:
    """查询备份数据。
    
    参数说明：
    ----------
    backup_file : str
        备份文件路径
    date_range : Optional[Tuple[int, int]]
        日期范围过滤，格式为(start_date, end_date)
    codes : Optional[List[str]]
        股票代码过滤列表
    storage_format : str
        存储格式，支持"json", "binary", "memory_map"
        
    返回值：
    -------
    NDArray[np.object_]
        查询结果数组，每行格式为[date, code, timestamp, *facs]
        注意：查询结果包含timestamp列
        
    示例：
    -------
    >>> backup_data = query_backup(
    ...     "results.bin",
    ...     date_range=(20220101, 20220131),
    ...     codes=['000001', '000002'],
    ...     storage_format="binary"
    ... )
    >>> print(backup_data[0])  # [date, code, timestamp, fac1, fac2, fac3]
    """
    ...

def delete_backup(backup_file: str, storage_format: str) -> None:
    """删除备份文件。
    
    参数说明：
    ----------
    backup_file : str
        备份文件路径
    storage_format : str
        存储格式
        
    示例：
    -------
    >>> delete_backup("results.bin", "binary")
    """
    ...

def backup_exists(backup_file: str, storage_format: str) -> bool:
    """检查备份文件是否存在。
    
    参数说明：
    ----------
    backup_file : str
        备份文件路径
    storage_format : str
        存储格式
        
    返回值：
    -------
    bool
        文件是否存在
        
    示例：
    -------
    >>> exists = backup_exists("results.bin", "binary")
    >>> print(exists)
    True
    """
    ...

def get_backup_info(backup_file: str, storage_format: str) -> Tuple[int, str]:
    """获取备份文件信息。
    
    参数说明：
    ----------
    backup_file : str
        备份文件路径
    storage_format : str
        存储格式
        
    返回值：
    -------
    Tuple[int, str]
        文件大小（字节）和修改时间
        
    示例：
    -------
    >>> size, modified_time = get_backup_info("results.bin", "binary")
    >>> print(f"文件大小: {size} 字节, 修改时间: {modified_time}")
    """
    ...