import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 定义颜色映射
COLORS = {
    1: '#ADD8E6',  # MIC Oxacillin (Light Blue)
    2: '#FDD017',  # MIC Shikimic acid (Gold/Yellow)
    3: '#104E8B',  # FIC (Dark Blue) - Adjusted for better visibility
    4: '#D3D3D3',  # No growth (Light Gray)
    5: '#CD5C5C',  # Growth (Indian Red / Pinkish)
}

# 手动映射的 8x12 数据矩阵 (基于提供的 Grid 数据)
# Rows A-H, Cols 1-12
plate_data = [
    # 1  2  3  4  5  6  7  8  9  10 11 12
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4], # A
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4], # B
    [4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 5, 4], # C
    [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 4], # D
    [4, 4, 4, 4, 3, 5, 5, 5, 5, 5, 5, 4], # E
    [4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 4], # F
    [4, 4, 1, 5, 5, 5, 5, 5, 5, 5, 5, 4], # G
    [4, 4, 1, 5, 5, 5, 5, 5, 5, 5, 5, 4], # H
]

def draw_well_plate(data, output_filename):
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 设置坐标轴范围和比例
    ax.set_xlim(-1.5, 13.5)
    ax.set_ylim(-2, 11)
    ax.set_aspect('equal')
    ax.axis('off') # 关闭默认坐标轴

    # 1. 绘制孔板背景 (圆角矩形)
    # FancyBboxPatch (x, y) 是左下角
    plate_rect = patches.FancyBboxPatch(
        (0.5, 0.5), 12, 8,
        boxstyle="round,pad=0.4",
        linewidth=2,
        edgecolor='#B0B0d0',
        facecolor='#F8F8FF', # GhostWhite
        zorder=0
    )
    ax.add_patch(plate_rect)
    
    # 内部凹陷效果 (可选，模拟图中的内框)
    inner_rect = patches.FancyBboxPatch(
        (0.6, 0.6), 11.8, 7.8,
        boxstyle="round,pad=0.2",
        linewidth=1,
        edgecolor='#D0D0E0',
        facecolor='none',
        zorder=1
    )
    ax.add_patch(inner_rect)

    rows = 8
    cols = 12
    row_labels = ['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'] 
    
    # 2. 绘制 96 个孔
    for r in range(rows):
        for c in range(cols):
            # 获取数据值
            value = data[r][c] 
            color = COLORS.get(value, 'white')
            
            # 坐标换算 (行 A 在最上面，对应 y=8; 列 1 在左边，对应 x=1)
            x = c + 1
            y = rows - r 
            
            # 绘制圆形孔
            # 增加一点阴影效果/边缘
            circle = patches.Circle(
                (x, y), 0.38, 
                linewidth=1.2, 
                edgecolor='#8888AA', 
                facecolor=color, 
                zorder=10
            )
            ax.add_patch(circle)
            
            # 绘制孔的编号 (A1, A12 etc) - 原图只有行列标，孔内无字

    # 3. 添加行标签 (A-H)
    for i, label in enumerate(row_labels):
        # i=0(H) -> y=1, i=7(A) -> y=8
        ax.text(0, i + 1, label, ha='center', va='center', fontsize=11, color='#808090', fontweight='bold')

    # 4. 添加列标签 (1-12)
    # Reverting to top as per original schematic, or bottom as per user's "reconstructed" image?
    # The user provided a "reconstructed" image which has labels at bottom.
    # But the reference schematic has them at top.
    # The user's prompt "use matplotlib to redraw this antimicrobial synergy 96-well plate" refers to the first image (schematic).
    # The schematic has numbers 1-12 at top.
    # So I should put them back to top, and legend closer to bottom.
    for i in range(cols):
        ax.text(i + 1, 8.8, str(i + 1), ha='center', va='center', fontsize=11, color='#808090', fontweight='bold')

    # 5. 绘制梯度指示器 (Shikimic acid 和 Oxacillin)
    
    # 顶部 Oxacillin 标题
    ax.text(6.5, 10.2, "Oxacillin", ha='center', va='center', fontsize=16, color='black')
    
    # 顶部梯度条 (三角/梯形)
    # 用多边形模拟渐变条形状: 左厚右薄
    top_gradient_poly = patches.Polygon(
        [(1, 9.4), (1, 9.8), (12.5, 9.4)], # 简单的楔形
        closed=True, 
        edgecolor='none',
        facecolor='#CCCCFF', # 淡紫/蓝
        alpha=0.6
        # 如果需要真正的渐变色，matplotlib 比较复杂，这里用纯色形状示意
    )
    ax.add_patch(top_gradient_poly)

    # 左侧 Shikimic acid 标题
    ax.text(-1.2, 4.5, "Shikimic acid", rotation=90, ha='center', va='center', fontsize=16, color='black')

    # 左侧梯度条
    # 下薄上厚
    left_gradient_poly = patches.Polygon(
        [(-0.4, 1), (-0.4, 8.5), (-0.8, 8.5)], 
        closed=True,
        edgecolor='none',
        facecolor='#CCCCFF',
        alpha=0.6
    )
    ax.add_patch(left_gradient_poly)

    # 6. 添加自定义图例
    # 使用 Line2D 来模拟圆孔，这样在图例中显示为圆形
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='No growth',
               markerfacecolor=COLORS[4], markeredgecolor='#8888AA', markersize=15, markeredgewidth=1.2),
        Line2D([0], [0], marker='o', color='w', label='Growth',
               markerfacecolor=COLORS[5], markeredgecolor='#8888AA', markersize=15, markeredgewidth=1.2),
        Line2D([0], [0], marker='o', color='w', label='MIC of Oxacillin',
               markerfacecolor=COLORS[1], markeredgecolor='#8888AA', markersize=15, markeredgewidth=1.2),
        Line2D([0], [0], marker='o', color='w', label='MIC of Shikimic acid',
               markerfacecolor=COLORS[2], markeredgecolor='#8888AA', markersize=15, markeredgewidth=1.2),
        Line2D([0], [0], marker='o', color='w', label='FIC',
               markerfacecolor=COLORS[3], markeredgecolor='#8888AA', markersize=15, markeredgewidth=1.2),
    ]

    # 将图例放在底部, 紧贴Plate
    legend = ax.legend(
        handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.125), 
        ncol=3, 
        frameon=False, 
        fontsize=11,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=1.5,
        borderaxespad=0.2
    )

    # 保存文件
    plt.tight_layout()
    # dpi=900, format='tif', compression logic handled by PIL usually, but matplotlib savefig supports basic params
    plt.savefig(output_filename, format='tiff', dpi=900, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    print(f"Successfully saved vector graphic to {output_filename}")

if __name__ == "__main__":
    draw_well_plate(plate_data, "plate_reconstruction.tif")
