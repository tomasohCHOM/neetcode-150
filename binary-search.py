from collections import defaultdict
from typing import List
import math


class Solution:
    # Binary Search (Easy)
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] > target:
                r = m - 1
            elif nums[m] < target:
                l = m + 1
            else:
                return m
        return -1

    # Search a 2D Matrix (Medium)
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        M, N = len(matrix), len(matrix[0])
        l, r = 0, len(matrix) * len(matrix[0]) - 1
        while l <= r:
            m = (l + r) // 2
            num = matrix[m // N][m % N]
            if num > target:
                r = m - 1
            elif num < target:
                l = m + 1
            else:
                return True
        return False

    # Koko Eating Bananas (Medium)
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        if len(piles) == h:
            return max(piles)

        def helper(k: int):
            count = 0
            for pile in piles:
                count += math.ceil(pile / k)
            return count

        l, r = 1, max(piles)
        output = 0
        while l <= r:
            m = (l + r) // 2
            if helper(m) > h:
                l = m + 1
            else:
                output = m
                r = m - 1
        return output

    # Find Minimum in Rotated Sorted Array (Medium)
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        output = float("inf")
        while l <= r:
            m = (l + r) // 2
            output = min(output, nums[m])
            if nums[m] > nums[r]:
                l = m + 1
            else:
                r = m - 1
        return output

    # Search in Rotated Sorted Array
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                return m
            if nums[l] > nums[m]:
                if target <= nums[r] and target > nums[m]:
                    l = m + 1
                else:
                    r = m - 1
            else:
                if target >= nums[l] and target < nums[m]:
                    r = m - 1
                else:
                    l = m + 1
        return -1

    # Time Based Key Value Store
    class TimeMap:
        def __init__(self):
            # dictionary of list items with entries (time, value)
            self.map = defaultdict(list)

        def set(self, key: str, value: str, timestamp: int) -> None:
            self.map[key].append((timestamp, value))

        def get(self, key: str, timestamp: int) -> str:
            if not self.map[key]:
                return ""
            output, values = "", self.map[key]
            l, r = 0, len(self.map[key]) - 1
            while l <= r:
                m = (l + r) // 2
                if values[m][0] > timestamp:
                    r = m - 1
                elif values[m][0] < timestamp:
                    output = values[m][1]
                    l = m + 1
                else:
                    return values[m][1]
            return output

    # Median of Two Sorted Arrays
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def solve(A: List[int], B: List[int]) -> float:
            total = len(A) + len(B)
            half = total // 2
            # A is guaranteed to be smaller
            l, r = 0, len(A) - 1
            # There will always be an answer because m + n >= 1
            while True:
                # Get the index of the left partition side of A
                i = (l + r) // 2
                # Get the index of the left partition side of B
                j = half - i - 2
                # Retrieve numbers at those indices, as well as the next numbers
                A_prev = A[i] if i >= 0 else float("-inf")
                A_next = A[i + 1] if (i + 1) < len(A) else float("inf")
                B_prev = B[j] if j >= 0 else float("-inf")
                B_next = B[j + 1] if (j + 1) < len(B) else float("inf")

                # Case 1: We found our left partition
                if B_next >= A_prev and A_next >= B_prev:
                    # Odd case: just return the next element
                    if total % 2 == 1:
                        return min(A_next, B_next)
                    # Even case: return (m1 + m2) / 2
                    return (max(A_prev, B_prev) + min(A_next, B_next)) / 2
                # Case 2: Our partition from A is too large, so decrease A
                elif A_prev > B_next:
                    r = i - 1
                # Case 3: Our partition from B is too large, so increase A
                else:
                    l = i + 1

        if len(nums1) > len(nums2):
            return solve(nums2, nums1)
        return solve(nums1, nums2)
