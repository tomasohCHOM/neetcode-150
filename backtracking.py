from typing import List


class Solution:
    # Subsets (Medium)
    def subsets(self, nums: List[int]) -> List[List[int]]:
        output = []

        def dfs(i, subset):
            if i >= len(nums):
                output.append(subset.copy())
                return
            subset.append(nums[i])
            dfs(i + 1, subset)  # include
            subset.pop()
            dfs(i + 1, subset)  # NOT include

        dfs(0, [])
        return output

    # Combination Sum (Medium)
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        output = []

        def dfs(i, subset, total):
            if i >= len(candidates) or total > target:
                return
            if total == target:
                output.append(subset.copy())
                return
            subset.append(candidates[i])
            dfs(i, subset, total + candidates[i])
            subset.pop()
            dfs(i + 1, subset, total)

        dfs(0, [], 0)
        return output

    # Permutations (Medium)
    def permute(self, nums: List[int]) -> List[List[int]]:
        if len(nums) == 0:
            return [[]]
        perms = self.permute(nums[1:])
        output = []
        for p in perms:
            for i in range(len(p) + 1):
                p_copy = p.copy()
                p_copy.insert(i, nums[0])
                output.append(p_copy)
        return output

    # Subsets II (Medium)
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        output = []
        nums.sort()

        def dfs(i, subset):
            if i >= len(nums):
                output.append(subset.copy())
                return
            subset.append(nums[i])
            dfs(i + 1, subset)
            subset.pop()
            while i + 1 < len(nums) and nums[i] == nums[i + 1]:
                i += 1
            dfs(i + 1, subset)

        dfs(0, [])
        return output

    # Combination Sum II (Medium)
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        output = []

        def dfs(i, nums, total):
            if total == target:
                output.append(nums.copy())
                return
            if i >= len(candidates) or total > target:
                return
            nums.append(candidates[i])
            dfs(i + 1, nums, total + candidates[i])
            nums.pop()
            while i + 1 < len(candidates) and candidates[i] == candidates[i + 1]:
                i += 1
            dfs(i + 1, nums, total)

        dfs(0, [], 0)
        return output

    # Word Search (Medium)
    def exist(self, board: List[List[str]], word: str) -> bool:
        M, N = len(board), len(board[0])

        def backtrack(i, r, c, seen):
            if i == len(word):
                return True
            if (
                (r < 0 or r >= M)
                or (c < 0 or c >= N)
                or ((r, c) in seen)
                or (board[r][c] != word[i])
            ):
                return False
            seen.add((r, c))
            for r1, c1 in [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]]:
                if backtrack(i + 1, r1, c1, seen):
                    return True
            seen.remove((r, c))
            return False

        for r in range(M):
            for c in range(N):
                if backtrack(0, r, c, set()):
                    return True
        return False

    # Palindrome Partitioning (Medium)
    def partition(self, s: str) -> List[List[str]]:
        output = []

        def backtrack(i, sub):
            if i >= len(s):
                output.append(sub.copy())
                return
            for j in range(i, len(s)):
                if self.is_palindrome(s, i, j):
                    sub.append(s[i : j + 1])
                    backtrack(j + 1, sub)
                    sub.pop()

        backtrack(0, [])
        return output

    def is_palindrome(self, s: str, l: int, r: int) -> bool:
        while l < r:
            if s[l] != s[r]:
                return False
            l, r = l + 1, r - 1
        return True

    # Letter Combinations of a Phone Number (Medium)
    def letterCombinations(self, digits: str) -> List[str]:
        mappings = {
            2: "abc",
            3: "def",
            4: "ghi",
            5: "jkl",
            6: "mno",
            7: "pqrs",
            8: "tuv",
            9: "wxyz",
        }
        output = []

        def backtrack(i, curr):
            if i == len(digits):
                if curr:
                    output.append("".join(curr))
                return
            digit = int(digits[i])
            for letter in mappings[digit]:
                curr.append(letter)
                backtrack(i + 1, curr)
                curr.pop()

        backtrack(0, [])
        return output

    # N Queens (Hard)
    def solveNQueens(self, n: int) -> List[List[str]]:
        cols = set()
        pos_diag = set()
        neg_diag = set()
        board = [["."] * n for r in range(n)]
        output = []

        def backtrack(r):
            if r == len(board):
                print(board)
                output.append(["".join(row) for row in board])
                return
            for c in range(n):
                if (
                    c not in cols
                    and (r + c) not in pos_diag
                    and (r - c) not in neg_diag
                ):
                    cols.add(c)
                    pos_diag.add(r + c)
                    neg_diag.add(r - c)
                    board[r][c] = "Q"

                    backtrack(r + 1)

                    cols.remove(c)
                    pos_diag.remove(r + c)
                    neg_diag.remove(r - c)
                    board[r][c] = "."

        backtrack(0)
        return output
