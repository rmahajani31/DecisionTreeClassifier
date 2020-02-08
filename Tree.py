class Tree:

    def __init__(self):
        self.head = Node()



class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.col = -1

    def add_child(self, n):
        self.children.append(n)