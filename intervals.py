import heapq
from typing import List


# Used in Meeting Rooms I & II
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end


class Solution:
    # Insert Interval (Medium)
    def insert(
        self, intervals: List[List[int]], newInterval: List[int]
    ) -> List[List[int]]:
        N, i = len(intervals), 0
        output = []
        while i < N and intervals[i][1] < newInterval[0]:
            output.append(intervals[i])
            i += 1
        while i < N and newInterval[1] >= intervals[i][0]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1
        output.append(newInterval)
        while i < N:
            output.append(intervals[i])
            i += 1
        return output

    # Merge Intervals (Medium)
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        output = []
        for interval in intervals:
            if not output or interval[0] > output[-1][1]:
                output.append([interval[0], interval[1]])
            else:
                output[-1][1] = max(output[-1][1], interval[1])
        return output

    # Non Overlapping Intervals (Medium)
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        prev_end, output = float("-inf"), 0
        for start, end in intervals:
            if prev_end > start:
                output += 1
                prev_end = min(prev_end, end)
            else:
                prev_end = end
        return output

    # Meeting Rooms (Easy)
    def canAttendMeetings(self, intervals: List[Interval]) -> bool:
        intervals.sort(key=lambda x: x.start)
        prev_time = None
        for interval in intervals:
            if prev_time and prev_time.end > interval.start:
                return False
            prev_time = interval
        return True

    # Meeting Rooms II (Medium)
    def minMeetingRooms(self, intervals: List[Interval]) -> int:
        intervals.sort(key=lambda x: x.start)
        pq = []
        for interval in intervals:
            if pq and interval.start >= pq[0]:
                heapq.heappop(pq)
            heapq.heappush(pq, interval.end)
        return len(pq)

    # Minimum Interval to Include Each Query (Hard)
    def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
        intervals.sort()
        pq, output, i = [], {}, 0
        for q in sorted(queries):
            while i < len(intervals) and intervals[i][0] <= q:
                l, r = intervals[i]
                heapq.heappush(pq, (r - l + 1, r))
                i += 1
            while pq and pq[0][1] < q:
                heapq.heappop(pq)
            output[q] = pq[0][0] if pq else -1
        return [output[q] for q in queries]
