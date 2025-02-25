# # Node class representing each node in the Binary Search Tree (BST)
# class Node:
#     def __init__(self, key):
#         self.data = key       # The value stored in the node
#         self.left = None      # Pointer to the left child
#         self.right = None     # Pointer to the right child
#
# # Binary Search Tree class
# class BST:
#     def __init__(self):
#         self.root = None  # Initialize an empty tree
#
#     # Public method to insert a key into the BST
#     def insert(self, key):
#         if not self.root:  # If the tree is empty, create the root node
#             self.root = Node(key)
#         else:  # Otherwise, call the helper function to insert recursively
#             self._insert(self.root, key)
#
#     # Helper method to handle recursive insertion
#     def _insert(self, root, key):
#         if key < root.data:  # If key is less than current node's value
#             if not root.left:
#                 root.left = Node(key)  # Insert as left child
#             else:
#                 self._insert(root.left, key)  # Recur for left subtree
#         elif key > root.data:  # If key is greater than current node's value
#             if not root.right:
#                 root.right = Node(key)  # Insert as right child
#             else:
#                 self._insert(root.right, key)  # Recur for right subtree
#
#     # Public method to search for a key in the BST
#     def search(self, key):
#         return self._search(self.root, key)  # Call the helper method
#
#     # Helper method to handle recursive search
#     def _search(self, root, key):
#         if not root or root.data == key:  # Base case: Node found or tree is empty
#             return root
#         # Recur for left or right subtree based on the key
#         return self._search(root.left if key < root.data else root.right, key)
#
#     # Public method to perform in-order traversal
#     def inorder(self):
#         return self._inorder(self.root)  # Call the helper method
#
#     # Helper method for recursive in-order traversal
#     def _inorder(self, root):
#         if not root:
#             return []  # Base case: return empty list for empty subtree
#         # Recur for left subtree, visit current node, then right subtree
#         return self._inorder(root.left) + [root.data] + self._inorder(root.right)
#
# # Main program
# # Create a Binary Search Tree instance
# bst = BST()
#
# # Insert multiple values into the BST
# values = [6, 9, 5, 2, 8, 15, 24, 14, 7, 8, 5, 2]
# for v in values:
#     bst.insert(v)
#
# # Perform in-order traversal and print the result
# print("In-order traversal:", bst.inorder())
#
# # Search for the element 8 and print the result
# if bst.search(8):
#     print("Element 8 found.")
# else:
#     print("Element 8 not found.")

























def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# Example usage
arr = [10, 7, 8, 9, 1, 5]
sorted_arr = quick_sort(arr)
print("Sorted array:", sorted_arr)











