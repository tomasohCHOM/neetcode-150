from typing import List


class Solution:
    # Climbing Stairs (Easy)
    def climbStairs(self, n: int) -> int:
        if n <= 3:
            return n
        first, second = 3, 2
        for _ in range(4, n + 1):
            temp = first
            first = first + second
            second = temp
        return first

    # Min Cost Climbing Stairs (Easy)
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        first, second = cost[1], cost[0]
        for i in range(2, len(cost)):
            first, second = cost[i] + min(first, second), first
        return min(first, second)

    # House Robber (Medium)
    def rob(self, nums: List[int]) -> int:
        first, second = 0, 0
        for num in nums:
            temp = first
            first = max(num + second, first)
            second = temp
        return first

    # House Robber II (Medium)
    def rob(self, nums: List[int]) -> int:
        def helper(l, r):
            first, second = 0, 0
            for i in range(l, r):
                temp = first
                first = max(nums[i] + second, first)
                second = temp
            return first

        return max(nums[0], helper(1, len(nums)), helper(0, len(nums) - 1))

    # Longest Palindromic Substring (Medium)
    def longestPalindrome(self, s: str) -> str:
        def checkPalindrome(l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l, r = l - 1, r + 1
            return s[l + 1 : r]

        output = ""
        for i in range(len(s)):
            # Odd length
            candidate = checkPalindrome(i, i)
            if len(candidate) > len(output):
                output = candidate
            # Even length
            candidate = checkPalindrome(i, i + 1)
            if len(candidate) > len(output):
                output = candidate
        return output

    # Palindromic Substrings (Medium)
    def countSubstrings(self, s: str) -> int:
        def countPalindromes(l, r):
            count = 0
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l, r = l - 1, r + 1
                count += 1
            return count

        output = 0
        for i in range(len(s)):
            output += countPalindromes(i, i) + countPalindromes(i, i + 1)
        return output

    # Decode Ways (Medium)
    def numDecodings(self, s: str) -> int:
        first, second, third = 0, 1, 0
        for i in range(len(s) - 1, -1, -1):
            if s[i] != "0":
                first += second
                if i + 1 < len(s) and int(s[i : i + 2]) <= 26:
                    first += third
            first, second, third = 0, first, second
        return second

    # Coin Change (Medium)
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0
        for x in range(1, amount + 1):
            for c in coins:
                if x - c >= 0:
                    dp[x] = min(dp[x], 1 + dp[x - c])
        return dp[amount] if dp[amount] != amount + 1 else -1

    # Maximum Product Subarray (Medium)
    def maxProduct(self, nums: List[int]) -> int:
        curr_max, curr_min, output = nums[0], nums[0], nums[0]
        for i in range(1, len(nums)):
            prev_max, prev_min = curr_max, curr_min
            curr_max = max(nums[i], nums[i] * prev_min, nums[i] * prev_max)
            curr_min = min(nums[i], nums[i] * prev_min, nums[i] * prev_max)
            output = max(output, curr_max)
        return output

    # Word Break (Medium)
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        wordDict = set(wordDict)
        dp = [False] * (len(s) + 1)
        dp[0] = True
        for j in range(1, len(s) + 1):
            for i in range(j):
                if dp[i] and s[i:j] in wordDict:
                    dp[j] = True
                    break
        return dp[-1]

    # Longest Increasing Subsequence (Medium)
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1 for _ in range(len(nums))]
        for i in range(len(nums) - 1, -1, -1):
            for j in range(i, len(nums)):
                if nums[i] < nums[j]:
                    dp[i] = max(dp[i], 1 + dp[j])
        return max(dp)

    # Partition Equal Subset Sum (Medium)
    def canPartition(self, nums: List[int]) -> bool:
        if sum(nums) % 2 != 0:
            return False
        target = sum(nums) // 2
        dp = set()
        dp.add(0)
        for num in nums:
            if num > target:
                return False
            newDp = set()
            for t in dp:
                if t + num == target:
                    return True
                newDp.add(num + t)
                newDp.add(t)
            dp = newDp
        return False
