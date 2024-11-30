from typing import List


# Implement Trie Prefix Tree (Medium)
class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur_node = self.root
        for c in word:
            idx = ord(c) - ord("a")
            if not cur_node.children[idx]:
                cur_node.children[idx] = TrieNode()
            cur_node = cur_node.children[idx]
        cur_node.end = True

    def search(self, word: str) -> bool:
        cur_node = self.root
        for c in word:
            idx = ord(c) - ord("a")
            if not cur_node.children[idx]:
                return False
            cur_node = cur_node.children[idx]
        return cur_node.end

    def startsWith(self, prefix: str) -> bool:
        cur_node = self.root
        for c in prefix:
            idx = ord(c) - ord("a")
            if not cur_node.children[idx]:
                return False
            cur_node = cur_node.children[idx]
        return True


# Design Add And Search Words Data Structure (Medium)
class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        cur_node = self.root
        for c in word:
            idx = ord(c) - ord("a")
            if not cur_node.children[idx]:
                cur_node.children[idx] = TrieNode()
            cur_node = cur_node.children[idx]
        cur_node.end = True

    def search(self, word: str) -> bool:
        def dfs(i, node):
            cur_node = node
            for j in range(i, len(word)):
                c = word[j]
                if c == ".":
                    for child in cur_node.children:
                        if not child:
                            continue
                        if dfs(j + 1, child):
                            return True
                    return False
                idx = ord(c) - ord("a")
                if not cur_node.children[idx]:
                    return False
                cur_node = cur_node.children[idx]
            return cur_node.end

        return dfs(0, self.root)


# Word Seach II (Hard)
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.end = False

            def add_word(self, word):
                curr_node = self
                for c in word:
                    if c not in curr_node.children:
                        curr_node.children[c] = TrieNode()
                    curr_node = curr_node.children[c]
                curr_node.end = True

        trie = TrieNode()
        for word in words:
            trie.add_word(word)

        M, N = len(board), len(board[0])
        output, seen = set(), set()

        def dfs(r, c, curr_node, curr_str):
            if (
                min(r, c) < 0
                or r >= M
                or c >= N
                or (r, c) in seen
                or board[r][c] not in curr_node.children
            ):
                return

            curr_node = curr_node.children[board[r][c]]
            curr_str += board[r][c]
            if curr_node.end:
                output.add(curr_str)

            seen.add((r, c))
            dfs(r + 1, c, curr_node, curr_str)
            dfs(r - 1, c, curr_node, curr_str)
            dfs(r, c + 1, curr_node, curr_str)
            dfs(r, c - 1, curr_node, curr_str)
            seen.remove((r, c))

        for r in range(M):
            for c in range(N):
                dfs(r, c, trie, "")

        return list(output)
