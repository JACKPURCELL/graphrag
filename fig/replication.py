import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer
import matplotlib
plt.rcParams['font.size'] = 14
matplotlib.rcParams['font.family'] = 'optima'
matplotlib.rcParams['font.weight'] = 'medium'

watermark_types = ["john23","xuandong23b","aiwei23","xiaoniu23","rohith23","aiwei23b","scott22"]
thresholds = {
    "john23": 2.358025378986253,
    "xuandong23b": 2.545584412271571,
    "aiwei23": 24.998546600341797,
    "rohith23": 1.8526251316070557,
    "xiaoniu23": 0.00,
    "lean23": 0.984638512134552,
    "scott22": 0.17697394677108003,
    "aiwei23b": 0.2496753585975497
}
watermark_colors = {
    "rohith23": "orange",
    "xuandong23b": "deepskyblue",
    "john23": "limegreen",
    "aiwei23": "purple",
    "xiaoniu23": "magenta",
    "aiwei23b": "red",
    "scott22": "royalblue"
}
replace_dict = {
    "john23": "TGRL",
    "xuandong23b": "UG",
    "aiwei23": "UPV",
    "rohith23": "RDF",
    "xiaoniu23": "UB",
    "scott22": "GO",
    "aiwei23b": "SIR",
}
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
fig, ax = plt.subplots(figsize=(7, 4))

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 添加这一行来定义颜色列表

for i, watermark_type in enumerate(watermark_types):
    file_path = f"/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/opt/truncated/gen_table_w_metrics.jsonl"
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            try:
                length = data["w_wm_output_length"]
            except:
                length = len(tokenizer(data["w_wm_output"])['input_ids'])
            data_list.append(json.loads(line))

    # Initialize variables to track the token lengths and the corresponding TPRs
    token_lengths = range(10, 201, 10)  # Token lengths from 10 to 200, inclusive, in steps of 10
    true_positives = [0] * 20  # To store the count of True values for each token length range
    total_counts = [0] * 20  # To store the total count of predictions for each token length range

    # Process the data to calculate the TPR for each token length range
    for data in data_list:
        # Determine the index for the token_lengths based on the output length
        index = min(max((data["w_wm_output_length"] - 1) // 10, 0), 19)
        # Update the counts
        true_positives[index] += 1 if data["w_wm_output_z_score"] > thresholds[watermark_type]  else 0
        total_counts[index] += 1

    # Calculate the TPR for each token length range
    tprs = [tp / total if total else 0 for tp, total in zip(true_positives, total_counts)]

    # Plotting
    ax.plot(token_lengths, tprs, color=watermark_colors[watermark_type], linestyle='-', marker='o', markersize=5, label=f'{replace_dict[watermark_type]}',linewidth=2)

ax.set_xlabel('Token Length')
ax.set_ylabel('TPR (with FPR = 0.01)')
# ax.set_title('TPR by Token Length for Watermarked Outputs')
ax.grid(True, linestyle='--')

ax.legend(ncol=2)
plt.ylim(0, 1.05)
# for y in np.arange(0, 1.05, 0.2):
#     ax.axhline(y, color='gray', linewidth=0.5, linestyle='--', zorder=0)

plt.xticks(range(20, 201, 20))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)

plt.tight_layout()
plt.savefig('./plot/newplot/output/tpr_token.pdf')

print('./plot/newplot/output/tpr_token.pdf')

# import matplotlib.pyplot as plt
# import numpy as np
# import math
# import json
# from sklearn.metrics import roc_curve, auc
# from transformers import AutoTokenizer
# plt.rcParams['font.size'] = 14  # 设置全局字体大小为14
# # watermark_types = ["john23","xuandong23b","aiwei23","lean23","rohith23","aiwei23b","xiaoniu23"]


# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
# fig, ax = plt.subplots(figsize=(12, 7))

# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 添加这一行来定义颜色列表

# file_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/john23/c4/opt/truncated/gen_table_w_metrics.jsonl"
# data_list = []
# with open(file_path, 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         try:
#             length = data["w_wm_output_length"]
#         except:
#             length = len(tokenizer(data["w_wm_output"])['input_ids'])
#         data_list.append(json.loads(line))
        
        

# # Initialize variables to track the token lengths and the corresponding TPRs
# token_lengths = range(10, 201, 10)  # Token lengths from 10 to 200, inclusive, in steps of 10
# true_positives = [0] * 20  # To store the count of True values for each token length range
# total_counts = [0] * 20  # To store the total count of predictions for each token length range

# # Process the data to calculate the TPR for each token length range
# for data in data_list:
#     # Determine the index for the token_lengths based on the output length
#     index = min(max((data["w_wm_output_length"] - 1) // 10, 0), 19)
#     # Update the counts
#     true_positives[index] += 1 if data["w_wm_output_prediction"] else 0
#     total_counts[index] += 1

# # Calculate the TPR for each token length range
# tprs = [tp / total if total else 0 for tp, total in zip(true_positives, total_counts)]

# # Plotting
# fig, ax = plt.subplots(figsize=(12, 7))
# ax.plot(token_lengths, tprs, color='b', linestyle='-', marker='o', markersize=5, label='TPR vs. Token Length')
# ax.set_xlabel('Token Length')
# ax.set_ylabel('True Positive Rate (TPR)')
# ax.set_title('TPR by Token Length for Watermarked Outputs')
# ax.grid(True)

# ax.legend()
# plt.ylim(0, 1.05)
# plt.xticks(range(20, 201, 20))
# plt.savefig('plot/tpr_token.pdf')