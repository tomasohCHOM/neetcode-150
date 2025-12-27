import collections
from typing import List


class Solution:
    # Rotate Image (Medium)
    def rotate(self, matrix: List[List[int]]) -> None:
        N = len(matrix)
        for r in range(N):
            for c in range(r + 1):
                matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]
        for r in range(N):
            matrix[r].reverse()

    # Spiral Matrix (Medium)
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        left, right = 0, len(matrix[0])
        top, bottom = 0, len(matrix)
        output = []
        while left < right and top < bottom:
            for i in range(left, right):
                output.append(matrix[top][i])
            top += 1
            for i in range(top, bottom):
                output.append(matrix[i][right - 1])
            right -= 1
            if not (top < bottom and left < right):
                break
            for i in range(right - 1, left - 1, -1):
                output.append(matrix[bottom - 1][i])
            bottom -= 1
            for i in range(bottom - 1, top - 1, -1):
                output.append(matrix[i][left])
            left += 1
        return output

    # Set Matrix Zeroes (Medium)
    def setZeroes(self, matrix: List[List[int]]) -> None:
        M, N = len(matrix), len(matrix[0])
        row_set, col_set = [], []
        for r in range(M):
            for c in range(N):
                if not matrix[r][c]:
                    row_set.append(r)
                    col_set.append(c)
        for r in row_set:
            for c in range(N):
                matrix[r][c] = 0
        for c in col_set:
            for r in range(M):
                matrix[r][c] = 0

    # Happy Number (Easy)
    def isHappy(self, n: int) -> bool:
        def get_digit_squares(x):
            sqrs = 0
            while x > 0:
                sqrs += (x % 10) ** 2
                x //= 10
            return sqrs

        slow, fast = n, get_digit_squares(n)
        while slow != fast:
            fast = get_digit_squares(get_digit_squares(fast))
            slow = get_digit_squares(slow)
        return True if fast == 1 else False

    # Plus One (Easy)
    def plusOne(self, digits: List[int]) -> List[int]:
        curr = len(digits) - 1
        while curr >= 0:
            if digits[curr] != 9:
                digits[curr] += 1
                return digits
            digits[curr] = 0
            curr -= 1
        digits[0] = 1
        digits.append(0)
        return digits

    # Pow(x, n) (Medium)
    def myPow(self, x: float, n: int) -> float:
        output, power = 1, abs(n)
        while power != 0:
            if power & 1:
                output *= x
            x *= x
            power >>= 1
        return output if n >= 0 else 1 / output

    # Multiply Strings (Medium)
    def multiply(self, num1: str, num2: str) -> str:
        if "0" in [num1, num2]:
            return "0"

        num1, num2 = num1[::-1], num2[::-1]
        M, N = len(num1), len(num2)
        output = [0] * (M + N)
        for i in range(M):
            for j in range(N):
                prod = int(num2[j]) * int(num1[i])
                output[i + j] += prod
                output[i + j + 1] += output[i + j] // 10
                output[i + j] = output[i + j] % 10

        output, i = output[::-1], 0
        while i < len(output) and output[i] == 0:
            i += 1
        output = map(str, output[i:])
        return "".join(output)

    # Detect Squares (Medium)
    class DetectSquares:
        def __init__(self):
            self.points = []
            self.point_freq = collections.defaultdict(int)

        def add(self, point: List[int]) -> None:
            self.points.append(point)
            self.point_freq[tuple(point)] += 1

        def count(self, point: List[int]) -> int:
            output = 0
            ix, iy = point
            for x, y in self.points:
                if (abs(iy - y) != abs(ix - x)) or ix == x or iy == y:
                    continue
                output += self.point_freq[(x, iy)] * self.point_freq[(ix, y)]
            return output
