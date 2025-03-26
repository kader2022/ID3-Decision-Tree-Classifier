from collections import Counter
from Node import Node, Leaf
import json

class ID3Tree:
    def __init__(self, dataset=None):
        self.dataset = dataset
        self.tree = None
    
    def build(self):
        if self.dataset is None or len(self.dataset) == 0:
            raise ValueError("Dataset cannot be empty!")
        self.tree = self._build_tree(self.dataset)
        return self.tree
    
    def _build_tree(self, dataset):
        if len(dataset) == 0:
            raise ValueError("The dataset cannot be empty!")
        
        # If all examples have the same label, return a leaf
        if dataset.entropy(dataset.number_of_labels()) == 0:
            return Leaf(dataset.examples[0].label)
        
        # If no attributes left or only the class attribute, return leaf with most common label
        if len(dataset.attribute_names) <= 1:
            return Leaf(max(dataset.possible_labels(), key=lambda lbl: len(dataset.subset_by_label(lbl))))
        
        # Choose the best attribute
        best_attribute = dataset.optimal_attribute()
        if best_attribute is None:
            return Leaf(max(dataset.possible_labels(), key=lambda lbl: len(dataset.subset_by_label(lbl))))
        
        # Create a node for the best attribute
        root = Node(best_attribute)
        
        # For each value of the best attribute
        for value in dataset.possible_values_for_attribute(best_attribute):
            # Create a subset of examples where the attribute has this value
            subset = dataset.subset_by_attribute_value(best_attribute, value)
            
            # If subset is empty, add a leaf with the most common label from the parent dataset
            if not subset.examples:
                root.children[value] = Leaf(max(dataset.possible_labels(), 
                                              key=lambda lbl: len(dataset.subset_by_label(lbl))))
            else:
                # Recursively build the tree for this subset
                root.children[value] = self._build_tree(subset)
        
        return root
    
    def export_to_json(self, filename):
        """Export the tree to a JSON file"""
        if self.tree is None:
            raise ValueError("Tree has not been built yet!")
        
        tree_dict = self.tree.to_dict()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(tree_dict, f, indent=2, ensure_ascii=False)
    
    def get_tree_as_json_string(self):
        """Get the tree as a JSON string"""
        if self.tree is None:
            raise ValueError("Tree has not been built yet!")
        
        tree_dict = self.tree.to_dict()
        return json.dumps(tree_dict, indent=2, ensure_ascii=False)
    
    def classify(self, example):
        """Classify a single example using the tree"""
        if self.tree is None:
            raise ValueError("Tree has not been built yet!")
        
        node = self.tree
        while isinstance(node, Node):
            attribute = node.attribute_tested
            value = example.attributes.get(attribute)
            
            # If the attribute value is not in the tree, use the most common branch
            if value not in node.children or node.children[value] is None:
                # Find most frequent branch or leaf
                return self._most_frequent_label(node).label
            
            node = node.children[value]
        
        # When we reach a leaf, return its label
        return node.label
    
    def _most_frequent_label(self, node):
        """Find the most frequent label in the subtree"""
        label_counts = Counter()
        
        def count_labels(n):
            if isinstance(n, Leaf):
                label_counts[n.label] += 1
            else:
                for child in n.children.values():
                    if child is not None:
                        count_labels(child)
        
        count_labels(node)
        
        # Return the most common label as a Leaf
        if not label_counts:
            return Leaf("unknown")
        return Leaf(label_counts.most_common(1)[0][0])

    def get_tree_text_representation(self):
        """Get a text representation of the tree"""
        if self.tree is None:
            return "Tree has not been built yet!"
        
        lines = []
        self._get_tree_text(self.tree, "", True, lines)
        return "\n".join(lines)
    
    def _get_tree_text(self, node, prefix, is_tail, lines):
        """Recursively build a text representation of the tree"""
        if isinstance(node, Leaf):
            lines.append(f"{prefix}{'└── ' if is_tail else '├── '}[Label: {node.label}]")
            return
        
        # This is a decision node
        lines.append(f"{prefix}{'└── ' if is_tail else '├── '}[{node.attribute_tested}]")
        
        # Sort children for consistent display
        children = list(node.children.items())
        if not children:
            return
        
        # Process all but the last child
        for i, (value, child) in enumerate(children[:-1]):
            if child is not None:
                new_prefix = prefix + ('    ' if is_tail else '│   ')
                lines.append(f"{prefix}{'    ' if is_tail else '│   '}├── Value: {value}")
                self._get_tree_text(child, new_prefix, False, lines)
        
        # Process the last child
        value, child = children[-1]
        if child is not None:
            new_prefix = prefix + '    '
            lines.append(f"{prefix}{'    ' if is_tail else '│   '}└── Value: {value}")
            self._get_tree_text(child, new_prefix, True, lines)

