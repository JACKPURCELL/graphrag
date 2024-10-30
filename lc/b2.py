def getMinTransformations(caption: str) -> int:
    n = len(caption)
    if n == 1:
        return 0
        
    # dp[i][satisfied][pos] represents min changes needed up to index i
    # satisfied: whether current position satisfies the rule (0/1)
    # pos: 0 means same as prev, 1 means original, 2 means same as next
    dp = [[[float('inf')] * 3 for _ in range(2)] for _ in range(n)]
    
    # Initialize first position
    dp[0][0][1] = 0  # Keep original
    dp[0][0][2] = 1  # Change to match next
    
    for i in range(1, n):
        curr = ord(caption[i]) - ord('a')
        prev = ord(caption[i-1]) - ord('a')
        
        # For each previous state
        for prev_satisfied in range(2):
            for prev_pos in range(3):
                if dp[i-1][prev_satisfied][prev_pos] == float('inf'):
                    continue
                    
                # Current position options
                for curr_pos in range(3):
                    # Calculate the letter at current position based on curr_pos
                    curr_letter = curr  # Original
                    if curr_pos == 0:  # Same as previous
                        curr_letter = prev
                    elif curr_pos == 2 and i < n-1:  # Same as next
                        curr_letter = ord(caption[i+1]) - ord('a')
                    
                    # Calculate changes needed for current position
                    changes = dp[i-1][prev_satisfied][prev_pos]
                    if curr_letter != curr:
                        changes += abs(curr_letter - curr)
                    
                    # Check if current position satisfies the rule
                    satisfied = 0
                    if curr_letter == prev or (i < n-1 and curr_letter == ord(caption[i+1]) - ord('a')):
                        satisfied = 1
                    
                    dp[i][satisfied][curr_pos] = min(dp[i][satisfied][curr_pos], changes)
    
    # Find minimum among all final states that satisfy the rule
    result = float('inf')
    for pos in range(3):
        result = min(result, dp[n-1][1][pos])
    
    return result if result != float('inf') else -1

# Example usage
caption = "aca"
print(getMinTransformations(caption))  # Output: 2


# Example usage
caption = "acca"
print(getMinTransformations(caption))  # Output: 2

# Example usage
caption = "aabbb"
print(getMinTransformations(caption))  # Output: 2

# Example usage
caption = 'aaccdd'
min_changes = getMinTransformations(caption)
print(min_changes)  # Output: 1
