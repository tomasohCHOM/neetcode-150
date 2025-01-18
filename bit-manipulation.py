from typing import List
import math


class Solution:
    # Single Number (Easy)
    def singleNumber(self, nums: List[int]) -> int:
        output = 0
        for num in nums:
            output ^= num
        return output

    # Number of 1 Bits (Easy)
    def hammingWeight(self, n: int) -> int:
        output = 0
        for i in range(32):
            output += 1 if (1 << i) & n else 0
        return output

    # Counting Bits (Easy)
    def countBits(self, n: int) -> List[int]:
        output = []
        for i in range(n + 1):
            output.append(i.bit_count())
        return output

    # Reverse Bits (Easy)
    def reverseBits(self, n: int) -> int:
        output = 0
        for i in range(32):
            bit = (n >> i) & 1
            output |= bit << (31 - i)
        return output

    # Missing Number (Easy)
    def missingNumber(self, nums: List[int]) -> int:
        output = len(nums)
        for i, num in enumerate(nums):
            output ^= i ^ num
        return output

    # Sum of Two Integers (Medium)
    def getSum(self, a: int, b: int) -> int:
        mask = 0xFFFFFFFF
        max_int = 0x7FFFFFFF
        while b != 0:
            carry = (a & b) << 1
            a = (a ^ b) & mask
            b = carry & mask
        return a if a <= max_int else ~(a ^ mask)

    # Reverse Integer (Medium)
    def reverse(self, x: int) -> int:
        MIN = -2147483648
        MAX = 2147483647
        output = 0
        while x:
            digit = int(math.fmod(x, 10))
            x = int(x / 10)
            if output > MAX // 10 or (output == MAX // 10 and digit > MAX % 10):
                return 0
            if output < MIN // 10 or (output == MIN // 10 and digit < MIN % 10):
                return 0
            output = (output * 10) + digit
        return output
