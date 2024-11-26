import heapq
from typing import List
from collections import defaultdict, deque


class Solution:
    # Reconstruct Itinerary (Hard)
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        graph = defaultdict(list)
        for from_i, to_i in tickets:
            graph[from_i].append(to_i)
        for k in graph.keys():
            graph[k].sort(reverse=True)
        output = []

        def dfs(k: str):
            while graph[k]:
                neighbor = graph[k].pop()
                dfs(neighbor)
            output.append(k)

        dfs("JFK")
        return output[::-1]

    # Min Cost to Connect All Points (Medium)
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        def manhattan(first, second):
            return abs(first[0] - second[0]) + abs(first[1] - second[1])

        N = len(points)
        seen = set()
        pq = [(0, (points[0][0], points[0][1]))]
        output = 0
        while len(seen) < N:
            w, (u, v) = heapq.heappop(pq)
            if (u, v) in seen:
                continue
            output += w
            seen.add((u, v))
            for point in points:
                if (point[0], point[1]) not in seen and point != (u, v):
                    heapq.heappush(pq, (manhattan((u, v), point), (point[0], point[1])))
        return output

    # Network Delay Time (Medium)
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        adj = defaultdict(list)
        for u, v, w in times:
            adj[u].append((w, v))
        pq = [(0, k)]
        seen = set()
        output = 0
        while pq:
            w, node = heapq.heappop(pq)
            if node in seen:
                continue
            seen.add(node)
            output = max(output, w)
            for w2, neighbor in adj[node]:
                if neighbor not in seen:
                    heapq.heappush(pq, (w + w2, neighbor))
        return output if len(seen) == n else -1

    # Swim In Rising Water (Hard)
    def swimInWater(self, grid: List[List[int]]) -> int:
        N = len(grid)
        seen = set()
        pq = [(grid[0][0], (0, 0))]

        while pq:
            t, (r, c) = heapq.heappop(pq)
            if (r, c) in seen:
                continue
            if r == N - 1 and c == N - 1:
                return t
            seen.add((r, c))
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    heapq.heappush(pq, (max(t, grid[nr][nc]), (nr, nc)))

    # Alien Dictionary (Hard)
    def foreignDictionary(self, words: List[str]) -> str:
        adj = {c: set() for w in words for c in w}
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            min_len = min(len(w1), len(w2))
            if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
                return ""
            for j in range(min_len):
                if w1[j] != w2[j]:
                    adj[w1[j]].add(w2[j])
                    break

        seen, output = {}, []

        def dfs(char):
            if char in seen:
                return seen[char]
            seen[char] = True
            for n in adj[char]:
                if dfs(n):
                    return True
            seen[char] = False
            output.append(char)

        for c in adj:
            if dfs(c):
                return ""
        return "".join(output[::-1])

    # Cheapest Flights Within K Stops (Medium)
    def findCheapestPrice(
        self, n: int, flights: List[List[int]], src: int, dst: int, k: int
    ) -> int:
        prices = [float("inf")] * n
        prices[src] = 0
        adj = [[] for _ in range(n)]
        for u, v, w in flights:
            adj[u].append((v, w))

        q = deque([(0, src, 0)])
        while q:
            cost, node, stops = q.popleft()
            if stops > k:
                continue
            for neighbor, w in adj[node]:
                next_cost = cost + w
                if next_cost < prices[neighbor]:
                    prices[neighbor] = next_cost
                    q.append((next_cost, neighbor, stops + 1))
        return prices[dst] if prices[dst] != float("inf") else -1
