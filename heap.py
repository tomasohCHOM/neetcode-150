import heapq
from typing import List
from math import sqrt
from collections import Counter, deque, defaultdict


class Solution:
    # Kth Largest Element In a Stream (Easy)
    class KthLargest:
        def __init__(self, k: int, nums: List[int]):
            self.k = k
            self.min_heap = nums
            heapq.heapify(self.min_heap)
            while len(self.min_heap) > self.k:
                heapq.heappop(self.min_heap)

        def add(self, val: int) -> int:
            heapq.heappush(self.min_heap, val)
            while len(self.min_heap) > self.k:
                heapq.heappop(self.min_heap)
            return self.min_heap[0]

    # Last Stone Weight (Easy)
    def lastStoneWeight(self, stones: List[int]) -> int:
        stones = [(-1 * stone) for stone in stones]
        heapq.heapify(stones)
        while len(stones) > 1:
            first = -1 * heapq.heappop(stones)
            second = -1 * heapq.heappop(stones)
            if first != second:
                heapq.heappush(stones, -1 * (first - second))
        return -1 * stones[0] if stones else 0

    # K Closest Points to Origin (Medium)
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        distances = []
        for i, point in enumerate(points):
            distances.append((sqrt(point[0] ** 2 + point[1] ** 2), i))
        heapq.heapify(distances)
        output = []
        for i in range(k):
            output.append(points[heapq.heappop(distances)[1]])
        return output

    # Kth Largest Element In An Array (Medium)
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums_heap = [-1 * num for num in nums]
        heapq.heapify(nums_heap)
        for _ in range(k - 1):
            heapq.heappop(nums_heap)
        return -1 * heapq.heappop(nums_heap)

    # Task Scheduler (Medium)
    def leastInterval(self, tasks: List[str], n: int) -> int:
        freq = Counter(tasks)
        pq = [-v for v in freq.values()]
        heapq.heapify(pq)
        idle = deque()
        output = 0
        while pq or idle:
            output += 1
            if not pq:
                output = idle[0][1]
            else:
                task = 1 + heapq.heappop(pq)
                if task:
                    idle.append((task, output + n))
            if idle and idle[0][1] == output:
                heapq.heappush(pq, idle.popleft()[0])
        return output

    # Design Twitter (Medium)
    class Twitter:
        def __init__(self):
            self.count = 0
            # user_id -> set of followee_id
            self.follow_map = defaultdict(set)
            # user_id -> list of (count, tweet_ids)
            self.tweet_map = defaultdict(list)

        def postTweet(self, user_id: int, tweet_id: int) -> None:
            self.tweet_map[user_id].append((self.count, tweet_id))
            self.count -= 1

        def getNewsFeed(self, user_id: int) -> List[int]:
            pq, output = [], []
            self.follow_map[user_id].add(user_id)

            for followee_id in self.follow_map[user_id]:
                if followee_id in self.tweet_map:
                    last_index = len(self.tweet_map[followee_id]) - 1
                    count, tweet_id = self.tweet_map[followee_id][last_index]
                    heapq.heappush(pq, (count, tweet_id, followee_id, last_index - 1))

            while pq and len(output) < 10:
                count, tweet_id, followee_id, index = heapq.heappop(pq)
                output.append(tweet_id)
                if index >= 0:
                    count, tweet_id = self.tweet_map[followee_id][index]
                    heapq.heappush(pq, (count, tweet_id, followee_id, index - 1))
            return output

        def follow(self, follower_id: int, followee_id: int) -> None:
            self.follow_map[follower_id].add(followee_id)

        def unfollow(self, follower_id: int, followee_id: int) -> None:
            if followee_id in self.follow_map[follower_id]:
                self.follow_map[follower_id].remove(followee_id)

    # Find Median From Data Stream (Hard)
    class MedianFinder:
        def __init__(self):
            self.small, self.large = [], []

        def addNum(self, num: int) -> None:
            if self.large and num > self.large[0]:
                heapq.heappush(self.large, num)
            else:
                heapq.heappush(self.small, -1 * num)

            if len(self.small) > len(self.large) + 1:
                heapq.heappush(self.large, -1 * heapq.heappop(self.small))
            if len(self.large) > len(self.small) + 1:
                heapq.heappush(self.small, -1 * heapq.heappop(self.large))

        def findMedian(self) -> float:
            if len(self.small) > len(self.large):
                return -1 * self.small[0]
            if len(self.small) < len(self.large):
                return self.large[0]
            return ((-1 * self.small[0]) + self.large[0]) / 2.0
