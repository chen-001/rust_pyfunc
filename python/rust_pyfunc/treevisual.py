import graphviz
import pandas as pd
from IPython.display import display
from rust_pyfunc.rust_pyfunc import PriceTree

class PriceTreeViz:
    def __init__(self):
        self.tree = PriceTree()
    
    def build_tree(self, times, prices, volumes):
        """
        构建价格树
        """
        return self.tree.build_tree(times, prices, volumes)
    
    def get_tree_structure(self):
        """
        获取树结构
        """
        return self.tree.get_tree_structure()
    
    def visualize(self):
        """
        在Jupyter Notebook中可视化价格树结构
        """
        # 创建有向图
        dot = graphviz.Digraph(comment='Price Tree')
        dot.attr(rankdir='TB')
        
        # 获取节点和边的信息
        nodes, edges = self.tree.get_visualization_data()
        
        # 添加节点
        for node_id, label in nodes:
            dot.node(node_id, label)
        
        # 添加边
        for src, dst in edges:
            dot.edge(src, dst)
        
        # 在notebook中显示
        display(dot)
    
    def display_tree_stats(self):
        # 获取统计数据
        self.stats = self.tree.get_tree_statistics()
        
        df = pd.DataFrame(self.stats, columns=['Metric', 'Value'])
        return df.style.set_properties(**{
            'text-align': 'left',
            'padding': '10px',
            'border': '1px solid black'
        }).set_table_styles([
            {'selector': 'th',
            'props': [('background-color', '#f0f0f0'),
                    ('text-align', 'center'),
                    ('padding', '10px'),
                    ('border', '1px solid black')]},
            {'selector': 'caption',
            'props': [('caption-side', 'top'),
                    ('text-align', 'center'),
                    ('font-size', '1.2em'),
                    ('padding', '8px')]}
        ]).set_caption('Price Tree Statistics')
        
haha=1