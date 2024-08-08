from typing import List
from collections import defaultdict


class Solution:
    # Create a HashSet to store each number. If we try to insert an element that
    # is already there, then the array contains a duplicate.
    # Time: O(N), Space: O(N)
    def containsDuplicate(self, nums: List[int]) -> bool:
        num_set = set()
        for num in nums:
            if num in num_set:
                return True
            num_set.add(num)
        return False

    # All characters and frequencies of s must be the same as t. We can use a
    # frequency map for each and compare them at the end.
    # Time: O(N), Space: O(N)
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        count_s = defaultdict(int)
        count_t = defaultdict(int)
        for char_s, char_t in zip(s, t):
            count_s[char_s] += 1
            count_t[char_t] += 1
        return count_s == count_t

    # Find the target by subtracting the current number from it and see if we
    # can find a match from previously seen values (using a HashMap). Return the
    # indices from that match.
    # Time: O(N), Space: O(N)
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        prev_map = {}
        for i, num in enumerate(nums):
            diff = target - num
            if diff in prev_map:
                return [i, prev_map[diff]]
            prev_map[num] = i

    # The key here is that strs[i] consists only of lowercase English letters.
    # We can use an array of size 26, then group corresponding anagrams.
    # Time: O(M * N), where M is len(strs) and N is the avg length of a string.
    # Space: O(N)
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        freqs = defaultdict(list)
        for word in strs:
            freq = [0] * 26
            for char in word:
                freq[ord(char) - ord("a")] += 1
            freqs[tuple(freq)].append(word)
        return freqs.values()
