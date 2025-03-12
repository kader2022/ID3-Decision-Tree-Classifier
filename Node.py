from collections import defaultdict

class Node:
    def __init__(self, attribute):
        self.children = defaultdict(lambda: None)
        self.attribute_tested = attribute
    
    def __repr__(self):  
        return f"Node(attribute='{self.attribute_tested}', children={len(self.children)})"
    
    def to_dict(self):
        """Convert node to dictionary for JSON serialization"""
        result = {
            "type": "node",
            "attribute": self.attribute_tested,
            "children": {}
        }
        for value, child in self.children.items():
            if child is not None:
                result["children"][value] = child.to_dict()
        return result

class Leaf:
    def __init__(self, label):
        self.label = label
    
    def __repr__(self): 
        return f"Leaf(label='{self.label}')"
    
    def to_dict(self):
        """Convert leaf to dictionary for JSON serialization"""
        return {
            "type": "leaf",
            "label": self.label
        }
