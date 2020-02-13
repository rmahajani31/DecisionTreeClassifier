import numpy as np

class Tree:

    def __init__(self, n=None):
        self.head = n



class Node:
    def __init__(self, value="", col=-1, left_split_cats=np.array([]), right_split_cats=np.array([])):
        self.children = []
        self.value = value
        self.col = col
        self.left_split_cats = left_split_cats
        self.right_split_cats = right_split_cats

    def add_child(self, n):
        self.children.append(n)