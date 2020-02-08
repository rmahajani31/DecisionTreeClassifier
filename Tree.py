class Tree:

    def __init__(self, n=None):
        self.head = n



class Node:
    def __init__(self, value="", col=-1):
        self.children = []
        self.value = value
        self.col = col

    def add_child(self, n):
        self.children.append(n)