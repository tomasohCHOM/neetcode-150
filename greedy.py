from typing import List
from collections import Counter
import heapq


class Solution:
    # Maximum Subarray (Medium)
    def maxSubArray(self, nums: List[int]) -> int:
        output = nums[0]
        local_max = float("-inf")
        for num in nums:
            local_max = max(num, local_max + num)
            output = max(output, local_max)
        return output

    # Jump Game (Medium)
    def canJump(self, nums: List[int]) -> bool:
        reach = len(nums) - 1
        for i in range(len(nums) - 1, -1, -1):
            if i + nums[i] >= reach:
                reach = i
        return reach == 0

    # Jump Game II (Medium)
    def jump(self, nums: List[int]) -> int:
        output = 0
        l, r = 0, 0
        while r < len(nums) - 1:
            farthest = 0
            for i in range(l, r + 1):
                farthest = max(farthest, i + nums[i])
            l, r = r + 1, farthest
            output += 1
        return output

    # Gas Station (Medium)
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1
        gas_left, output = 0, 0
        for i in range(len(gas)):
            gas_left += gas[i] - cost[i]
            if gas_left < 0:
                gas_left = 0
                output = i + 1
        return output

    # Hand of Straights (Medium)
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        if len(hand) % groupSize != 0:
            return False
        count = Counter(hand)
        pq = list(count.keys())
        heapq.heapify(pq)
        while pq:
            first = pq[0]
            for i in range(first, first + groupSize):
                if i not in count:
                    return False
                count[i] -= 1
                if count[i] == 0:
                    if i != pq[0]:
                        return False
                    heapq.heappop(pq)
        return True

    # Merge Triplets to Form Target Triplet (Medium)
    def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
        x = y = z = False
        for t in triplets:
            if t[0] <= target[0] and t[1] <= target[1] and t[2] <= target[2]:
                x |= t[0] == target[0]
                y |= t[1] == target[1]
                z |= t[2] == target[2]
        return x and y and z

    # Partition Labels (Medium)
    def partitionLabels(self, s: str) -> List[int]:
        last = {c: i for i, c in enumerate(s)}
        output = []
        start, end = 0, 0
        for i, c in enumerate(s):
            end = max(end, last[c])
            if i == end:
                output.append(end - start + 1)
                start = i + 1
        return output

    # Valid Parenthesis String (Medium)
    def checkValidString(self, s: str) -> bool:
        cmax, cmin = 0, 0
        for c in s:
            if c == "(":
                cmax, cmin = cmax + 1, cmin + 1
            elif c == ")":
                cmax, cmin = cmax - 1, cmin - 1
            else:
                cmax, cmin = cmax + 1, cmin - 1
            if cmax < 0:
                return False
            cmin = max(cmin, 0)
        return cmin == 0
