#!/bin/bash

# 定义基础路径
base_path="/home/ljc/graphrag/ragtest/output"

# 找到排序最大的文件夹
latest_folder=$(ls -d $base_path/*/ | sort -r | head -n 1)

# 定义目标文件路径
target_file="$latest_folder/artifacts/merged_graph.graphml"

# 检查目标文件是否存在
if [ -f "$target_file" ]; then
    # 执行命令生成 PDF
    graphml2gv "$target_file" | dot -Tpdf -o merged_graph.graphml.pdf
    echo "PDF 文件已生成：merged_graph.graphml.pdf"
else
    echo "目标文件不存在：$target_file"
fi

