"""
This module implements a HashedLinkedList for caching purposes.
"""

class Node:
    """A node in a doubly linked list."""
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class HashedLinkedList:
    """
    A HashedLinkedList data structure for implementing an LRU cache.
    It combines a hash map (dictionary) for O(1) lookups and a doubly
    linked list to maintain the order of items for efficient eviction.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # Hash map for key -> node
        self.head = Node(0, 0)  # Dummy head
        self.tail = Node(0, 0)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add(self, node):
        """Add a node to the end of the linked list."""
        prev_node = self.tail.prev
        prev_node.next = node
        self.tail.prev = node
        node.prev = prev_node
        node.next = self.tail

    def get(self, key):
        """
        Get an item from the cache. Moves the accessed item to the end
        of the list to mark it as recently used.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.value
        return None

    def put(self, key, value):
        """
        Put an item in the cache. If the cache is full, it evicts the
        least recently used item.
        """
        if key in self.cache:
            self._remove(self.cache[key])
        
        node = Node(key, value)
        self._add(node)
        self.cache[key] = node
        
        if len(self.cache) > self.capacity:
            # Evict the least recently used item
            lru_node = self.head.next
            self._remove(lru_node)
            del self.cache[lru_node.key]

    def __len__(self):
        return len(self.cache)
