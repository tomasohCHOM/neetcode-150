from typing import List, Optional
from collections import defaultdict, deque


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class Solution:
    # Number of Islands (Medium)
    def numIslands(self, grid: List[List[str]]) -> int:
        M, N = len(grid), len(grid[0])
        output = 0
        seen = set()

        def dfs(r, c):
            if min(r, c) < 0 or r >= M or c >= N or (r, c) in seen or grid[r][c] == "0":
                return
            seen.add((r, c))
            for r1, c1 in [[r + 1, c], [r - 1, c], [r, c + 1], [r, c - 1]]:
                dfs(r1, c1)

        for r in range(M):
            for c in range(N):
                if (r, c) not in seen and grid[r][c] == "1":
                    dfs(r, c)
                    output += 1
        return output

    # Max Area of Island (Medium)
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        M, N = len(grid), len(grid[0])
        seen = set()

        def dfs(r, c):
            if min(r, c) < 0 or r >= M or c >= N or (r, c) in seen or not grid[r][c]:
                return 0
            seen.add((r, c))
            return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)

        output = 0
        for r in range(M):
            for c in range(N):
                output = max(output, dfs(r, c))
        return output

    # Clone Graph (Medium)
    def cloneGraph(self, node: Optional["Node"]) -> Optional["Node"]:
        seen = {}

        def dfs(node):
            if node in seen:
                return seen[node]
            copy = Node(node.val)
            seen[node] = copy
            for neighbor in node.neighbors:
                copy.neighbors.append(dfs(neighbor))
            return copy

        return dfs(node) if node else None

    # Walls And Gates - Islands and Treasure (Medium)
    def islandsAndTreasure(self, grid: List[List[int]]) -> None:
        M, N = len(grid), len(grid[0])

        def bfs(r, c):
            q = deque()
            q.append((r, c))
            seen = set()
            i = 0
            while q:
                print(q)
                for _ in range(len(q)):
                    x, y = q.popleft()
                    if (x, y) in seen:
                        continue
                    seen.add((x, y))
                    if grid[x][y] == -1:
                        continue
                    grid[x][y] = min(grid[x][y], i)
                    for r1, c1 in [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]:
                        if min(r1, c1) < 0 or r1 >= M or c1 >= N:
                            continue
                        q.append((r1, c1))
                i += 1

        for r in range(M):
            for c in range(N):
                if grid[r][c] == 0:
                    # Start BFS on this cell
                    bfs(r, c)

    # Rotting Oranges (Medium)
    def orangesRotting(self, grid: List[List[int]]) -> int:
        M, N = len(grid), len(grid[0])
        q = deque()
        fresh = 0
        for r in range(M):
            for c in range(N):
                if grid[r][c] == 1:
                    fresh += 1
                if grid[r][c] == 2:
                    q.append((r, c))

        output = 0
        while fresh and q:
            for _ in range(len(q)):
                x, y = q.popleft()
                for x1, y1 in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if min(x1, y1) < 0 or x1 >= M or y1 >= N or grid[x1][y1] != 1:
                        continue
                    grid[x1][y1] = 2
                    q.append((x1, y1))
                    fresh -= 1
            output += 1
        return output if not fresh else -1

    # Pacific Atlantic Water Flow (Medium)
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        M, N = len(heights), len(heights[0])
        pacific, atlantic = set(), set()

        def dfs(r, c, seen, prev_height):
            if (
                min(r, c) < 0
                or r >= M
                or c >= N
                or (r, c) in seen
                or heights[r][c] < prev_height
            ):
                return
            seen.add((r, c))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                dfs(r + dx, c + dy, seen, heights[r][c])

        for r in range(M):
            dfs(r, 0, pacific, heights[r][0])
            dfs(r, N - 1, atlantic, heights[r][N - 1])

        for c in range(N):
            dfs(0, c, pacific, heights[0][c])
            dfs(M - 1, c, atlantic, heights[M - 1][c])

        output = []
        for r in range(M):
            for c in range(N):
                if (r, c) in pacific and (r, c) in atlantic:
                    output.append((r, c))
        return output

    # Surrounded Regions (Medium)
    def solve(self, board: List[List[str]]) -> None:
        M, N = len(board), len(board[0])
        seen = set()

        def dfs(r, c):
            if (
                min(r, c) < 0
                or r >= M
                or c >= N
                or (r, c) in seen
                or board[r][c] == "X"
            ):
                return
            seen.add((r, c))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                dfs(r + dx, c + dy)

        for r in range(M):
            dfs(r, 0)
            dfs(r, N - 1)

        for c in range(N):
            dfs(0, c)
            dfs(M - 1, c)

        for r in range(M):
            for c in range(N):
                if (r, c) not in seen and board[r][c] == "O":
                    board[r][c] = "X"

    # Course Schedule (Medium)
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj = defaultdict(list)
        for course, prereq in prerequisites:
            adj[course].append(prereq)

        seen = defaultdict(int)

        def dfs(course):
            if seen[course] == -1:
                return False
            if seen[course] == 1:
                return True
            seen[course] = -1
            for neighbor in adj[course]:
                if not dfs(neighbor):
                    return False
            seen[course] = 1
            return True

        for num in range(numCourses):
            if not dfs(num):
                return False
        return True

    # Course Schedule II (Medium)
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        adj = defaultdict(list)
        for course, prereq in prerequisites:
            adj[course].append(prereq)

        seen = defaultdict(int)
        output = []

        def dfs(course):
            if seen[course] == -1:
                return False
            if seen[course] == 1:
                return True
            seen[course] = -1
            for neighbor in adj[course]:
                if not dfs(neighbor):
                    return False
            seen[course] = 1
            output.append(course)
            return True

        for num in range(numCourses):
            if not dfs(num):
                return []
        return output

    # Graph Valid Tree (Medium)
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        # tree -> undirected connected graph with no cycles
        # our goal will be to
        adj = defaultdict(list)
        for edge in edges:
            adj[edge[0]].append(edge[1])
            adj[edge[1]].append(edge[0])

        seen = set()

        def dfs(node, prev):
            if node in seen:
                return False

            seen.add(node)
            for neighbor in adj[node]:
                if prev == neighbor:
                    continue
                if not dfs(neighbor, node):
                    return False
            return True

        return dfs(0, -1) and len(seen) == n

    # Number of Connected Components In An Undirected Graph (Medium)
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        par = [i for i in range(n)]
        rank = [1] * n

        def find(n1):
            p = par[n1]
            while p != par[p]:
                par[p] = par[par[p]]
                p = par[p]
            return p

        def union(n1, n2):
            p1, p2 = find(n1), find(n2)
            if p1 == p2:
                return False

            if rank[p1] > rank[p2]:
                par[p2] = p1
                rank[p1] += rank[p2]
            else:
                par[p1] = p2
                rank[p2] += rank[p1]
            return True

        output = n
        for n1, n2 in edges:
            if union(n1, n2):
                output -= 1
        return output

    # Redundant Connection (Medium)
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        N = len(edges)
        parents = [i for i in range(N + 1)]
        rank = [1] * (N + 1)

        def find(n):
            if n == parents[n]:
                return parents[n]
            parents[n] = find(parents[n])
            return parents[n]
        
        def union(n1, n2):
            p1, p2 = find(n1), find(n2)
            if p1 == p2:
                return False
            if rank[p1] > rank[p2]:
                parents[p2] = p1
                rank[p1] += rank[p2]
            else:
                parents[p1] = p2
                rank[p2] += rank[p1]
            return True
        
        for u, v in edges:
            if not union(u, v):
                return [u, v]


    # Word Ladder (Hard)
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0

        # Construct adjacency list based on pattern
        adj = defaultdict(list)
        for word in wordList:
            for j in range(len(word)):
                pattern = word[:j] + "*" + word[j + 1 :]
                adj[pattern].append(word)

        # BFS Part
        q = deque([beginWord])
        seen = set([beginWord])
        output = 1
        while q:
            for _ in range(len(q)):
                word = q.popleft()
                if word == endWord:
                    return output
                for j in range(len(word)):
                    pattern = word[:j] + "*" + word[j + 1 :]
                    for neighbor in adj[pattern]:
                        if neighbor not in seen:
                            q.append(neighbor)
                            seen.add(neighbor)
            output += 1
        return 0
