from typing import Optional, List
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    # Invert Binary Tree (Easy)
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root

    # Maximum Depth of Binary Tree (Easy)
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def dfs(node, depth):
            if not node:
                return depth
            return max(dfs(node.left, depth + 1), dfs(node.right, depth + 1))

        return dfs(root, 0)

    # Diameter of Binary Tree (Easy)
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        output = 0

        def dfs(node):
            nonlocal output
            if not node:
                return -1
            left = dfs(node.left)
            right = dfs(node.right)
            output = max(output, left + right + 2)
            return max(left, right) + 1

        dfs(root)
        return output

    # Balanced Binary Tree (Easy)
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def dfs(node):
            if not node:
                return (True, 0)
            left = dfs(node.left)
            right = dfs(node.right)
            return (
                left[0] and right[0] and abs(left[1] - right[1]) <= 1,
                1 + max(left[1], right[1]),
            )

        return dfs(root)[0]

    # Same Tree (Easy)
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not (p and q):
            return False
        return (
            p.val == q.val
            and self.isSameTree(p.left, q.left)
            and self.isSameTree(p.right, q.right)
        )

    # Subtree of Another Tree (Easy)
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not root and not subRoot:
            return True
        if not root:
            return False
        if self.helper(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def helper(self, root, subRoot):
        if not root and not subRoot:
            return True
        if not (root and subRoot):
            return False
        return (
            root.val == subRoot.val
            and self.helper(root.left, subRoot.left)
            and self.helper(root.right, subRoot.right)
        )

    # Lowest Common Ancestor of a Binary Search Tree (Medium)
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        curr = root
        while curr:
            if p.val < curr.val and q.val < curr.val:
                curr = curr.left
            elif p.val > curr.val and q.val > curr.val:
                curr = curr.right
            else:
                return curr

    # Binary Tree Level Order Traversal (Medium)
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        q = deque()
        if root:
            q.append(root)
        output = []
        while q:
            output.append([])
            for _ in range(len(q)):
                node = q.popleft()
                output[-1].append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return output

    # Binary Tree Right Side View (Medium)
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        q = deque()
        if root:
            q.append(root)
        output = []
        while q:
            LEN = len(q)
            for i in range(LEN):
                node = q.popleft()
                if not i:
                    output.append(node.val)
                if node.right:
                    q.append(node.right)
                if node.left:
                    q.append(node.left)
        return output

    # Count Good Nodes In Binary Tree (Medium)
    def goodNodes(self, root: TreeNode) -> int:
        output = 0

        def dfs(node, max_value):
            nonlocal output
            if not node:
                return
            if node.val >= max_value:
                output += 1
            dfs(node.left, max(max_value, node.val))
            dfs(node.right, max(max_value, node.val))

        dfs(root, root.val)
        return output

    # Validate Binary Search Tree (Medium)
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        self.prev = None

        def inorder(node, prev):
            if not node:
                return True
            if not inorder(node.left, self.prev):
                return False
            if self.prev and node.val <= self.prev.val:
                return False
            self.prev = node
            return inorder(node.right, self.prev)

        return inorder(root, self.prev)

    # Kth Smallest Element In a Bst (Medium)
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        output, idx = [-1], [0]

        def inorder(node):
            if not node:
                return
            if idx[0] <= k:
                inorder(node.left)
                idx[0] += 1
                if idx[0] == k:
                    output[0] = node.val
                inorder(node.right)

        inorder(root)
        return output[0]

    # Construct Binary Tree From Preorder And Inorder Traversal (Medium)
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder:
            return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1 : mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1 :], inorder[mid + 1 :])
        return root

    # Binary Tree Maximum Path Sum (Hard)
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.res = float("-inf")

        def dfs(node):
            if not node:
                return 0
            left = max(dfs(node.left), 0)
            right = max(dfs(node.right), 0)
            self.res = max(self.res, node.val + left + right)
            return node.val + max(left, right)

        dfs(root)
        return self.res

    # Serialize And Deserialize Binary Tree (Hard)
    class Codec:
        def serialize(self, root):
            output = []

            def dfs(node):
                if not node:
                    output.append("#")
                    return
                output.append(str(node.val))
                dfs(node.left)
                dfs(node.right)

            dfs(root)
            return ",".join(output)

        def deserialize(self, data):
            nodes = data.split(",")
            self.i = 0

            def dfs():
                if nodes[self.i] == "#":
                    self.i += 1
                    return None
                node = TreeNode(int(nodes[self.i]))
                self.i += 1
                node.left = dfs()
                node.right = dfs()
                return node

            return dfs()
