"""
HMM趋势预测可视化工具
独立的可视化模块，可以直接导入使用
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any


def plot_hmm_prediction(prices: List[float], hmm_result, 
                       title: str = "HMM趋势预测分析",
                       width: int = 1200, height: int = 800,
                       show_signals: bool = True,
                       signal_threshold: float = 0.6) -> go.Figure:
    """
    绘制HMM趋势预测结果
    
    参数:
    ----
    prices : List[float]
        价格序列
    hmm_result : HMMPredictionResult
        HMM预测结果对象
    title : str
        图表标题
    width, height : int
        图表尺寸
    show_signals : bool
        是否显示交易信号
    signal_threshold : float
        信号阈值
        
    返回:
    ----
    go.Figure
        Plotly图表对象
    """
    
    # 数据准备
    n_prices = len(prices)
    n_predictions = len(hmm_result.state_predictions)
    
    price_indices = list(range(n_prices))
    prediction_indices = list(range(n_prices - n_predictions, n_prices))
    
    # 提取概率数据
    down_probs = [pred[0] for pred in hmm_result.state_predictions]
    sideways_probs = [pred[1] for pred in hmm_result.state_predictions]
    up_probs = [pred[2] for pred in hmm_result.state_predictions]
    
    # 创建子图
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            '价格序列与交易信号',
            '状态预测概率时间序列', 
            '预测状态与概率分布'
        ],
        row_heights=[0.45, 0.35, 0.20]
    )
    
    # === 第一行：价格序列 ===
    fig.add_trace(
        go.Scatter(
            x=price_indices,
            y=prices,
            mode='lines',
            name='价格',
            line=dict(color='black', width=2.5)
        ),
        row=1, col=1
    )
    
    # 添加交易信号
    if show_signals:
        buy_points_x, buy_points_y = [], []
        sell_points_x, sell_points_y = [], []
        
        for i, (up_prob, down_prob) in enumerate(zip(up_probs, down_probs)):
            idx = prediction_indices[i]
            price = prices[idx]
            
            if up_prob > signal_threshold:
                buy_points_x.append(idx)
                buy_points_y.append(price)
            elif down_prob > signal_threshold:
                sell_points_x.append(idx)
                sell_points_y.append(price)
        
        # 买入信号
        if buy_points_x:
            fig.add_trace(
                go.Scatter(
                    x=buy_points_x,
                    y=buy_points_y,
                    mode='markers',
                    name=f'买入信号 (>{signal_threshold:.1%})',
                    marker=dict(
                        size=12,
                        color='green',
                        symbol='triangle-up',
                        line=dict(color='white', width=2)
                    )
                ),
                row=1, col=1
            )
        
        # 卖出信号  
        if sell_points_x:
            fig.add_trace(
                go.Scatter(
                    x=sell_points_x,
                    y=sell_points_y,
                    mode='markers',
                    name=f'卖出信号 (>{signal_threshold:.1%})',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='triangle-down',
                        line=dict(color='white', width=2)
                    )
                ),
                row=1, col=1
            )
    
    # === 第二行：概率时间序列 ===
    fig.add_trace(
        go.Scatter(
            x=prediction_indices,
            y=down_probs,
            mode='lines',
            name='下跌概率',
            line=dict(color='red', width=2),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=prediction_indices,
            y=sideways_probs,
            mode='lines',
            name='震荡概率',
            line=dict(color='gray', width=2),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=prediction_indices,
            y=up_probs,
            mode='lines',
            name='上涨概率',
            line=dict(color='green', width=2),
            fill='tonexty',
            fillcolor='rgba(0,255,0,0.1)'
        ),
        row=2, col=1
    )
    
    # 添加信号阈值线
    if show_signals:
        fig.add_hline(
            y=signal_threshold, 
            line_dash="dash", 
            line_color="orange",
            annotation_text=f"信号阈值: {signal_threshold:.1%}",
            row=2, col=1
        )
    
    # === 第三行：预测状态分布 ===
    optimal_states = []
    state_colors = []
    
    for pred in hmm_result.state_predictions:
        max_idx = pred.index(max(pred))
        state = max_idx - 1
        optimal_states.append(state)
        state_colors.append({-1: 'red', 0: 'gray', 1: 'green'}[state])
    
    fig.add_trace(
        go.Scatter(
            x=prediction_indices,
            y=optimal_states,
            mode='markers',
            name='最优预测状态',
            marker=dict(
                size=6,
                color=state_colors,
                line=dict(color='black', width=0.5),
                opacity=0.8
            )
        ),
        row=3, col=1
    )
    
    # 添加概率强度的柱状图背景
    max_probs = [max(pred) for pred in hmm_result.state_predictions]
    fig.add_trace(
        go.Bar(
            x=prediction_indices,
            y=max_probs,
            name='预测置信度',
            marker_color=state_colors,
            opacity=0.3,
            showlegend=False
        ),
        row=3, col=1
    )
    
    # === 更新布局 ===
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18, color='darkblue')
        ),
        width=width,
        height=height,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left", 
            x=1.01,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    # 设置坐标轴
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="概率", range=[0, 1], row=2, col=1)
    fig.update_yaxes(
        title_text="状态/置信度",
        ticktext=['下跌', '震荡', '上涨'],
        tickvals=[-1, 0, 1],
        range=[-1.5, 1.5],
        row=3, col=1
    )
    fig.update_xaxes(title_text="时间点", row=3, col=1)
    
    return fig


def plot_transition_matrix_evolution(hmm_result, 
                                    sample_points: int = 10,
                                    title: str = "状态转移矩阵演化") -> go.Figure:
    """
    绘制状态转移矩阵的演化过程
    
    参数:
    ----
    hmm_result : HMMPredictionResult
        HMM预测结果对象
    sample_points : int
        采样点数量
    title : str
        图表标题
        
    返回:
    ----
    go.Figure
        Plotly图表对象
    """
    
    if not hmm_result.transition_probs:
        raise ValueError("没有状态转移概率数据")
    
    # 采样关键时间点
    n_steps = len(hmm_result.transition_probs)
    step_size = max(1, n_steps // sample_points)
    sample_indices = list(range(0, n_steps, step_size))
    
    state_names = ['下跌', '震荡', '上涨']
    
    # 创建子图矩阵 (3x3 for transition matrix)
    fig = make_subplots(
        rows=3, cols=3,
        shared_yaxes=True,
        subplot_titles=[f'{state_names[i//3]} → {state_names[i%3]}' 
                       for i in range(9)],
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )
    
    colors = ['red', 'gray', 'green']
    
    # 为每个转移概率创建时间序列
    for from_state in range(3):
        for to_state in range(3):
            row = from_state + 1
            col = to_state + 1
            
            # 提取该转移的概率时间序列
            prob_series = [hmm_result.transition_probs[i][from_state][to_state] 
                          for i in sample_indices]
            
            fig.add_trace(
                go.Scatter(
                    x=sample_indices,
                    y=prob_series,
                    mode='lines+markers',
                    name=f'{state_names[from_state]}→{state_names[to_state]}',
                    line=dict(color=colors[to_state], width=2),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        height=700,
        width=900
    )
    
    # 设置y轴范围
    for row in range(1, 4):
        for col in range(1, 4):
            fig.update_yaxes(range=[0, 1], row=row, col=col)
    
    return fig


def analyze_prediction_performance(prices: List[float], 
                                 hmm_result,
                                 lookback_periods: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
    """
    分析预测性能
    
    参数:
    ----
    prices : List[float]
        价格序列
    hmm_result : HMMPredictionResult
        HMM预测结果
    lookback_periods : List[int]
        分析的未来期数
        
    返回:
    ----
    Dict[str, Any]
        性能分析结果
    """
    
    n_prices = len(prices)
    n_predictions = len(hmm_result.state_predictions)
    prediction_start = n_prices - n_predictions
    
    results = {}
    
    for lookback in lookback_periods:
        correct_predictions = 0
        total_predictions = 0
        state_performance = {-1: {'correct': 0, 'total': 0},
                           0: {'correct': 0, 'total': 0},
                           1: {'correct': 0, 'total': 0}}
        
        for i, pred in enumerate(hmm_result.state_predictions):
            if prediction_start + i + lookback >= n_prices:
                break
            
            # 获取预测状态
            max_idx = pred.index(max(pred))
            predicted_state = max_idx - 1
            
            # 计算实际表现
            current_price = prices[prediction_start + i]
            future_price = prices[prediction_start + i + lookback]
            change_pct = (future_price / current_price - 1) * 100
            
            # 确定实际状态
            if change_pct > 2.0:  # 上涨阈值
                actual_state = 1
            elif change_pct < -2.0:  # 下跌阈值
                actual_state = -1
            else:
                actual_state = 0  # 震荡
            
            # 统计
            state_performance[predicted_state]['total'] += 1
            if predicted_state == actual_state:
                correct_predictions += 1
                state_performance[predicted_state]['correct'] += 1
            
            total_predictions += 1
        
        # 计算准确率
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # 计算各状态的精确率
        state_precision = {}
        for state in [-1, 0, 1]:
            total = state_performance[state]['total']
            correct = state_performance[state]['correct']
            state_precision[state] = correct / total if total > 0 else 0
        
        results[f'lookback_{lookback}'] = {
            'accuracy': overall_accuracy,
            'correct': correct_predictions,
            'total': total_predictions,
            'state_precision': state_precision
        }
    
    return results


def quick_hmm_plot(prices: List[float], hmm_result, save_filename: str = None) -> go.Figure:
    """
    快速绘制HMM结果的简化版本
    
    参数:
    ----
    prices : List[float]
        价格序列
    hmm_result : HMMPredictionResult  
        HMM预测结果
    save_filename : str, optional
        保存的HTML文件名
        
    返回:
    ----
    go.Figure
        Plotly图表对象
    """
    
    fig = plot_hmm_prediction(
        prices=prices,
        hmm_result=hmm_result,
        title="HMM趋势预测快速分析",
        show_signals=True,
        signal_threshold=0.5
    )
    
    if save_filename:
        fig.write_html(save_filename)
        print(f"📊 图表已保存为: {save_filename}")
    
    return fig


# 使用示例
def demo_usage():
    """
    演示如何使用可视化工具
    """
    print("📊 HMM可视化工具使用演示")
    
    # 这里需要实际的HMM预测结果
    print("请先运行HMM预测，然后使用以下方式调用:")
    print("""
    # 导入可视化工具
    from hmm_visualizer import plot_hmm_prediction, quick_hmm_plot
    
    # 执行HMM预测
    import rust_pyfunc
    result = rust_pyfunc.hmm_trend_prediction(prices=your_prices)
    
    # 创建可视化
    fig = plot_hmm_prediction(your_prices, result)
    fig.show()
    
    # 或者快速保存
    quick_hmm_plot(your_prices, result, "my_analysis.html")
    """)


if __name__ == "__main__":
    demo_usage()