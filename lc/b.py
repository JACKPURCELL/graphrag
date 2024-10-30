def min_changes_three_dim_dp(caption):
    n = len(caption)
    # 初始化 dp 数组: dp[i][j][k]
    # i: 当前处理到的字符位置
    # j: 0 - 未满足, 1 - 已满足
    # k: 0 - 前一个字母, 1 - 当前字母, 2 - 后一个字母
    dp = [[[float('inf')] * 3 for _ in range(2)] for _ in range(n + 1)]
    
    # 基础情况：空字符串已满足条件
    dp[0][1][1] = 0

    for i in range(1, n + 1):
        current_char = ord(caption[i - 1]) - ord('a')
        for prev_j in range(2):
            for prev_k in range(3):
                for k in range(3):
                    # 计算当前字母与目标字母（k）的差距
                    diff = abs(current_char - (k - 1))
                    if diff > 1:
                        continue  # 不可能的转移

                    # 如果前一个状态未满足，当前字母必须与前一个相同
                    if prev_j == 0 and k != prev_k:
                        continue

                    # 更新当前状态
                    new_j = 1 if k == prev_k else 0
                    dp[i][new_j][k] = min(dp[i][new_j][k], dp[i-1][prev_j][prev_k] + (1 if diff == 1 else 0))

    # 返回最后一个字符的最小修改次数
    return min(min(dp[n][j]) for j in range(2))


print(min_changes_three_dim_dp("acca"))  