from typing import List


class Solution:
    # Valid Parentheses (Easy)
    def isValid(self, s: str) -> bool:
        stack = []
        for char in s:
            if char == "[":
                stack.append("]")
            elif char == "{":
                stack.append("}")
            elif char == "(":
                stack.append(")")
            else:
                if not stack or stack[-1] != char:
                    return False
                stack.pop()
        return not stack

    # Min Stack (Medium)
    class MinStack:
        def __init__(self):
            self.stack = []
            self.min_elem = float("inf")

        def push(self, val: int) -> None:
            self.min_elem = min(self.min_elem, val)
            self.stack.append((val, self.min_elem))

        def pop(self) -> None:
            self.stack.pop()
            if self.stack:
                self.min_elem = self.getMin()
            else:
                self.min_elem = float("inf")

        def top(self) -> int:
            return self.stack[-1][0]

        def getMin(self) -> int:
            return self.stack[-1][1]

    # Evaluate Reverse Polish Notation
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        ops = set(["+", "-", "*", "/"])
        for token in tokens:
            if token in ops:
                first, second = stack.pop(), stack.pop()
                if token == "+":
                    op = second + first
                elif token == "-":
                    op = second - first
                elif token == "*":
                    op = second * first
                elif token == "/":
                    op = int(second / first)
                stack.append(op)
            else:
                stack.append(int(token))
        return stack.pop()

    # Generate Parentheses (Medium)
    def generateParenthesis(self, n: int) -> List[str]:
        output = []

        def backtrack(cur: str, open_count: int, close_count: int):
            if len(cur) == n * 2:
                output.append(cur)
            if open_count < n:
                backtrack(cur + "(", open_count + 1, close_count)
            if close_count < open_count:
                backtrack(cur + ")", open_count, close_count + 1)

        backtrack("", 0, 0)
        return output

    # Daily Temperatures (Medium)
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        st = []
        output = [0] * len(temperatures)
        for i, temp in enumerate(temperatures):
            while st and temperatures[st[-1]] < temp:
                prev = st.pop()
                output[prev] = i - prev
            st.append(i)
        return output

    # Car Fleet (Medium)
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        pairs = sorted([(p, s) for (p, s) in zip(position, speed)], reverse=True)
        st = []
        for p, s in pairs:
            steps = (target - p) / s
            if st and steps <= st[-1]:
                continue
            st.append(steps)
        return len(st)

    # Largest Rectangle in Histogram (Hard)
    def largestRectangleArea(self, heights: List[int]) -> int:
        # Monotonic increasing stack
        output = 0
        st = []
        for i, height in enumerate(heights):
            start = i
            while st and height < st[-1][0]:
                (prev_height, prev_idx) = st.pop()
                area = (i - prev_idx) * prev_height
                output = max(output, area)
                start = prev_idx
            st.append((height, start))

        # if elements still remaining in the stack
        while st:
            (height, prev_idx) = st.pop()
            area = (len(heights) - prev_idx) * height
            output = max(output, area)
        return output
