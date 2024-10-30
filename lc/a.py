def min_prefix_substring(idStream, videoIds):
    # 预处理 idStream，计算从开头到每个位置的字符计数
    n = len(idStream)
    count = [[0] * 10 for _ in range(n + 1)]  # 10 是因为字符是 '0'-'9'

    for i in range(1, n + 1):
        for d in range(10):
            count[i][d] = count[i - 1][d]
        count[i][int(idStream[i - 1])] += 1

    def can_form_target(right, target_count):
        # 检查从开头到 right 的前缀是否包含目标字符串的所有字符
        for d in range(10):
            if count[right][d] < target_count[d]:
                return False
        return True

    results = []
    for target in videoIds:
        target_count = [0] * 10
        for char in target:
            target_count[int(char)] += 1

        # 二分查找最小的 right
        left, right = 0, n
        found = False
        while left < right:
            mid = (left + right) // 2
            if can_form_target(mid, target_count):
                found = True
                right = mid
            else:
                left = mid + 1

        if found:
            results.append(right)
        else:
            results.append(-1)

    return results

# 示例输入
idStream = "064819848398"
videoIds = ["088", "364", "07"]
print(min_prefix_substring(idStream, videoIds))  # 输出: [7, 10, -1]
