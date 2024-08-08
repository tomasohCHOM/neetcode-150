from typing import Optional, List
import heapq


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Node:
    def __init__(self, x: int, next: "Node" = None, random: "Node" = None):
        self.val = int(x)
        self.next = next
        self.random = random


class LRUNode:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None


class Solution:
    # Reverse Linked List (Easy)
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev

    # Merge Two Sorted Lists (Easy)
    def mergeTwoLists(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> Optional[ListNode]:
        dummy = ListNode()
        curr = dummy
        while list1 and list2:
            if list1.val < list2.val:
                curr.next = list1
                list1 = list1.next
            else:
                curr.next = list2
                list2 = list2.next
            curr = curr.next
        if list1 or list2:
            curr.next = list1 if list1 else list2
        return dummy.next

    # Reorder List (Medium)
    def reorderList(self, head: Optional[ListNode]) -> None:
        # Find middle node
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # Reverse the second half of the list
        fast, prev = slow.next, None
        slow.next = None
        while fast:
            temp = fast.next
            fast.next = prev
            prev = fast
            fast = temp
        # The new head is stored in prev
        first, second = head, prev
        # Merge the two lists
        while second:
            temp1 = first.next
            temp2 = second.next
            first.next = second
            second.next = temp1
            first, second = temp1, temp2

    # Remove Nth Node From End of List (Medium)
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        slow, fast = dummy, head
        for _ in range(n):
            fast = fast.next
        while fast:
            fast, slow = fast.next, slow.next
        slow.next = slow.next.next
        return dummy.next

    # Copy List With Random Pointer (Medium)
    def copyRandomList(self, head: "Optional[Node]") -> "Optional[Node]":
        if not head:
            return None

        curr = head
        # Interweave nodes of old and copied list: A > A' > B > B' ...
        while curr:
            node = Node(curr.val, curr.next)
            curr.next = node
            curr = node.next
        # Assign random to copy nodes
        curr = head
        while curr:
            if curr.random:
                curr.next.random = curr.random.next
            curr = curr.next.next
        # Extract copied list
        new_head = head.next
        curr_old = head
        curr = new_head
        while curr_old:
            curr_old.next = curr_old.next.next
            curr.next = curr.next.next if curr.next else None
            curr, curr_old = curr.next, curr_old.next
        # Connect copy list nodes with each other
        return new_head

    # Add Two Numbers (Medium)
    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        carry = 0
        dummy = ListNode()
        curr = dummy
        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            curr_sum = v1 + v2 + carry
            carry = curr_sum // 10
            curr.next = ListNode(curr_sum % 10)
            curr = curr.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy.next

    # Linked List Cycle (Medium)
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

    # Find The Duplicate Number (Medium)
    def findDuplicate(self, nums: List[int]) -> int:
        slow = fast = 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        slow = 0
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        return slow

    # LRU Cache (Medium)
    class LRUCache:
        def __init__(self, capacity: int):
            self.capacity = capacity

            self.node_ptrs = {}
            self.head, self.tail = LRUNode(), LRUNode()
            self.head.next, self.tail.prev = self.tail, self.head

        def update(self, node) -> None:
            temp = node.next
            node.prev.next = temp
            temp.prev = node.prev

            temp = self.head.next
            temp.prev = node
            node.prev = self.head
            self.head.next = node
            node.next = temp

        def get(self, key: int) -> int:
            if not key in self.node_ptrs:
                return -1
            node = self.node_ptrs[key]
            self.update(node)
            return node.val

        def put(self, key: int, value: int) -> None:
            if key in self.node_ptrs:
                node = self.node_ptrs[key]
                node.val = value
                self.update(node)
                return

            if len(self.node_ptrs) == self.capacity:
                lru = self.tail.prev
                self.tail.prev = lru.prev
                lru.prev.next = self.tail
                del self.node_ptrs[lru.key]

            # Add the node to the cache
            new_node = LRUNode(key, value)
            temp = self.head.next
            temp.prev = new_node
            self.head.next = new_node
            new_node.next = temp
            new_node.prev = self.head
            self.node_ptrs[key] = new_node

    # Merge K Sorted Lists (Hard)
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        pq = []
        for i, node in enumerate(lists):
            if node:
                heapq.heappush(pq, (node.val, i, node))
        dummy = ListNode()
        curr = dummy
        while pq:
            val, i, node = heapq.heappop(pq)
            curr.next = node
            if node.next:
                heapq.heappush(pq, (node.next.val, i, node.next))
            curr = curr.next
        return dummy.next

    # Reverse Nodes In K Group (Hard)
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        prev_group = dummy
        while True:
            kth = self.getKth(prev_group, k)
            if not kth:
                break
            next_group = kth.next
            curr, prev = prev_group.next, kth.next
            while curr != next_group:
                temp = curr.next
                curr.next = prev
                prev = curr
                curr = temp
            temp = prev_group.next
            prev_group.next = kth
            prev_group = temp
        return dummy.next

    def getKth(self, curr, k):
        while curr and k > 0:
            curr = curr.next
            k -= 1
        return curr
