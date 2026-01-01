from typing import List
import collections
import functools


class Solution:
    # Unique Paths (Medium)
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n
        for _ in range(m - 2, -1, -1):
            for c in range(n - 2, -1, -1):
                dp[c] += dp[c + 1]
        return dp[0]

    # Longest Common Subsequence (Medium)
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        M, N = len(text1), len(text2)
        dp_i1 = [0] * (N + 1)
        for i in range(M - 1, -1, -1):
            dp_i = [0] * (N + 1)
            for j in range(N - 1, -1, -1):
                dp_i[j] = (
                    1 + dp_i1[j + 1]
                    if text1[i] == text2[j]
                    else max(dp_i1[j], dp_i[j + 1])
                )
            dp_i1 = dp_i
        return dp_i1[0]

    # Best Time to Buy And Sell Stock With Cooldown (Medium)
    def maxProfit(self, prices: List[int]) -> int:
        dp1_buy, dp1_sell, dp2_buy = 0, 0, 0
        for i in range(len(prices) - 1, -1, -1):
            dp_buy = max(dp1_sell - prices[i], dp1_buy)
            dp_sell = max(dp2_buy + prices[i], dp1_sell)
            dp2_buy = dp1_buy
            dp1_buy, dp1_sell = dp_buy, dp_sell
        return dp1_buy

    # Coin Change II (Medium)
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [1] + [0] * amount
        for c in coins:
            for x in range(c, amount + 1):
                dp[x] += dp[x - c]
        return dp[amount]

    # Target Sum (Medium)
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        dp_i1 = collections.defaultdict(int)
        dp_i1[target] = 1
        for num in nums:
            dp_i = collections.defaultdict(int)
            for curr_sum, count in dp_i1.items():
                dp_i[curr_sum + num] += count
                dp_i[curr_sum - num] += count
            dp_i1 = dp_i
        return dp_i1[0]

    # Interleaving String (Medium)
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        M, N, K = len(s1), len(s2), len(s3)
        if M + N != K:
            return False

        @functools.cache
        def dfs(i, j):
            if i + j == K:
                return i == M and j == N
            output = False
            if i < M and s1[i] == s3[i + j]:
                output = dfs(i + 1, j)
            if not output and j < N and s2[j] == s3[i + j]:
                output = dfs(i, j + 1)
            return output

        return dfs(0, 0)

    # Longest Increasing Path In a Matrix (Hard)
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        M, N = len(matrix), len(matrix[0])
        dp = [[0] * N for _ in range(M)]

        def dfs(r: int, c: int, prev: int) -> int:
            if r < 0 or c < 0 or r == M or c == N or matrix[r][c] <= prev:
                return 0
            if dp[r][c]:
                return dp[r][c]
            output = 1
            for nr, nc in [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]:
                output = max(output, 1 + dfs(nr, nc, matrix[r][c]))
            dp[r][c] = output
            return output

        output = 0
        for r in range(M):
            for c in range(N):
                output = max(output, dfs(r, c, -1))
        return output

    # Distinct Subsequences (Hard)
    def numDistinct(self, s: str, t: str) -> int:
        M, N = len(s), len(t)
        dp = [0] * N + [1]
        for i in range(M - 1, -1, -1):
            prev = 1
            for j in range(N - 1, -1, -1):
                curr = dp[j]
                if s[i] == t[j]:
                    curr += prev
                prev = dp[j]
                dp[j] = curr
        return dp[0]

    # Edit Distance (Medium)
    def minDistance(self, word1: str, word2: str) -> int:
        M, N = len(word1), len(word2)
        dp_i1 = [N - j for j in range(N + 1)]
        dp_i = [0] * (N + 1)
        for i in range(M - 1, -1, -1):
            dp_i[N] = M - i
            for j in range(N - 1, -1, -1):
                dp_i[j] = (
                    dp_i1[j + 1]
                    if word1[i] == word2[j]
                    else 1 + min(dp_i1[j], dp_i1[j + 1], dp_i[j + 1])
                )
            dp_i1 = list(dp_i)
        return dp_i1[0]

    # Burst Balloons (Hard)
    def maxCoins(self, nums: List[int]) -> int:
        nums = [1] + nums + [1]
        N = len(nums)
        dp = [[0] * (N) for _ in range(N)]
        for l in range(N - 1, 0, -1):
            for r in range(l, N - 1):
                for i in range(l, r + 1):
                    coins = nums[l - 1] * nums[i] * nums[r + 1]
                    coins += dp[l][i - 1] + dp[i + 1][r]
                    dp[l][r] = max(dp[l][r], coins)
        return dp[1][N - 2]

    # Regular Expression Matching (Hard)
    def isMatch(self, s: str, p: str) -> bool:
        M, N = len(s), len(p)

        @functools.cache
        def dfs(i, j):
            if j == N:
                return i == M
            is_match = i < M and (s[i] == p[j] or p[j] == ".")
            if (j + 1) < N and p[j + 1] == "*":
                return dfs(i, j + 2) or (is_match and dfs(i + 1, j))
            if is_match:
                return dfs(i + 1, j + 1)
            return False

        return dfs(0, 0)
