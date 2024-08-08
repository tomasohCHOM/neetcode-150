from typing import List
from collections import defaultdict


class Solution:
    # Contains Duplicate (Easy)
    def containsDuplicate(self, nums: List[int]) -> bool:
        num_set = set()
        for num in nums:
            if num in num_set:
                return True
            num_set.add(num)
        return False

    # Valid Anagram (Easy)
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        count_s = defaultdict(int)
        count_t = defaultdict(int)
        for char_s, char_t in zip(s, t):
            count_s[char_s] += 1
            count_t[char_t] += 1
        return count_s == count_t

    # Two Sum (Easy)
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        prev_map = {}
        for i, num in enumerate(nums):
            diff = target - num
            if diff in prev_map:
                return [i, prev_map[diff]]
            prev_map[num] = i

    # Group Anagrams (Medium)
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        freqs = defaultdict(list)
        for word in strs:
            freq = [0] * 26
            for char in word:
                freq[ord(char) - ord("a")] += 1
            freqs[tuple(freq)].append(word)
        return freqs.values()

    # Top K Frequent Elements (Medium)
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        max_count = 0
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
            max_count = max(max_count, freq[num])

        count = [[] for _ in range(max_count + 1)]
        for n, c in freq.items():
            count[c].append(n)

        output = []
        for i in range(len(count) - 1, 0, -1):
            for num in count[i]:
                output.append(num)
                if len(output) == k:
                    return output

    # Encode and Decode Strings (Medium) - two functions
    def encode(self, strs: List[str]) -> str:
        output = ""
        for s in strs:
            output += str(len(s)) + "#" + s
        return output

    def decode(self, s: str) -> List[str]:
        output = []
        i = 0
        while i < len(s):
            length = ""
            while s[i] != "#":
                length += s[i]
                i += 1
            length = int(length)
            output.append(s[i + 1 : i + length + 1])
            i = i + length + 1
        return output

    # Product of Array Except Self (Medium)
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        prefix = 1
        output = [1] * len(nums)
        for i in range(len(nums)):
            output[i] = prefix
            prefix *= nums[i]
        suffix = 1
        for i in range(len(nums) - 1, -1, -1):
            output[i] *= suffix
            suffix *= nums[i]
        return output

    # Valid Sudoku (Medium)
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = defaultdict(set)
        cols = defaultdict(set)
        squares = defaultdict(set)
        for r in range(len(board)):
            for c, value in enumerate(board[r]):
                if value == ".":
                    continue
                if (
                    value in rows
                    and r in rows[value]
                    or value in cols
                    and c in cols[value]
                    or value in squares
                    and (r // 3, c // 3) in squares[value]
                ):
                    return False
                rows[value].add(r)
                cols[value].add(c)
                squares[value].add((r // 3, c // 3))
        return True

    # Longest Consecutive Sequence (Medium)
    def longestConsecutive(self, nums: List[int]) -> int:
        output = 0
        num_set = set(nums)
        for num in num_set:
            if num - 1 not in num_set:
                seq = 1
                while num + seq in num_set:
                    seq += 1
                output = max(output, seq)
        return output
