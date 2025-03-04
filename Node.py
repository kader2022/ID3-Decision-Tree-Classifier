from collections import defaultdict

class Node:
    def __init__(self, attribute):
        self.children = defaultdict(lambda: None)
        self.attribute_tested = attribute

class Leaf:
    def __init__(self, label):
        self.label = label

