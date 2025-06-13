class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

class Tree:
    def __init__(self, data = 0):
        self.root = Node(data)
        
    