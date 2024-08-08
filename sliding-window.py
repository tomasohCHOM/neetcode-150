from typing import List
from collections import defaultdict, Counter, deque


class Solution:
    # Best Time to Buy And Sell Stock (Easy)
    def maxProfit(self, prices: List[int]) -> int:
        output = 0
        l = 0
        for r in range(len(prices)):
            profit = prices[r] - prices[l]
            if profit < 0:
                l = r
            output = max(output, profit)
        return output

    # Longest Substring Without Repeating Characters (Medium)
    def lengthOfLongestSubstring(self, s: str) -> int:
        chars, output = {}, 0
        l = 0
        for r in range(len(s)):
            if s[r] in chars and l <= chars[s[r]]:
                l = chars[s[r]] + 1
            chars[s[r]] = r
            output = max(output, r - l + 1)
        return output

    # Longest Repeating Character Replacement (Medium)
    def characterReplacement(self, s: str, k: int) -> int:
        freq = defaultdict(int)
        max_freq = 0
        l = 0
        for r in range(len(s)):
            freq[s[r]] += 1
            max_freq = max(max_freq, freq[s[r]])
            if (r - l + 1) - max_freq > k:
                freq[s[l]] -= 1
                l += 1
        return r - l + 1

    # Permutation In String (Medium)
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # Return immediately if s1 size > s2 size
        if len(s1) > len(s2):
            return False

        # For each character up until s1, count it in both
        s1Count, s2Count = [0] * 26, [0] * 26
        for i in range(len(s1)):
            s1Count[ord(s1[i]) - ord("a")] += 1
            s2Count[ord(s2[i]) - ord("a")] += 1

        # Count if there are matches for both s1 and s2
        matches = 0
        for i in range(26):
            matches += 1 if s1Count[i] == s2Count[i] else 0

        l = 0
        # Loop through the remaining characters
        for r in range(len(s1), len(s2)):
            # If they match, return True
            if matches == 26:
                return True

            # Calculate the new character. If it happens
            # to contribute to the anagram, increment matches
            # But if those characters already matched, decrement
            index = ord(s2[r]) - ord("a")
            s2Count[index] += 1
            if s1Count[index] == s2Count[index]:
                matches += 1
            elif s1Count[index] + 1 == s2Count[index]:
                matches -= 1

            # Calculate the old character (at idx l). If it happens
            # to contribute to the anagram, (removing it makes both
            # counts equal), increment matches. But if those characters
            # already matched before, decrement
            index = ord(s2[l]) - ord("a")
            s2Count[index] -= 1
            if s1Count[index] == s2Count[index]:
                matches += 1
            elif s1Count[index] - 1 == s2Count[index]:
                matches -= 1
            l += 1
        return matches == 26

    # Minimum Window Substring (Hard)
    def minWindow(self, s: str, t: str) -> str:
        if len(t) > len(s):
            return ""
        t_freq = Counter(t)
        chars = set(t)
        s_freq = defaultdict(int)
        cur_size = 0
        l, smallest = 0, float("inf")
        start, end = 0, 0
        for r in range(len(s)):
            if s[r] in t_freq:
                s_freq[s[r]] += 1
                if s_freq[s[r]] == t_freq[s[r]]:
                    cur_size += 1
            if r - l + 1 < len(t):
                continue
            while cur_size >= len(chars):
                if r - l + 1 < smallest:
                    smallest = min(smallest, r - l + 1)
                    start, end = l, r
                if s[l] in s_freq:
                    s_freq[s[l]] -= 1
                    if s_freq[s[l]] < t_freq[s[l]]:
                        cur_size -= 1
                l += 1
        if smallest == float("inf"):
            return ""
        return s[start : end + 1]

    # Sliding Window Maximum (Hard)
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        output = []
        q, l = deque(), 0
        for r in range(len(nums)):
            while q and nums[r] > nums[q[-1]]:
                q.pop()
            q.append(r)
            if q[0] < l:
                q.popleft()
            if r - l + 1 != k:
                continue
            output.append(nums[q[0]])
            l += 1
        return output
