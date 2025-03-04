from collections import Counter
from Data import Dataset
from Node import Node, Leaf

class ID3Tree:
    def __init__(self, file_path=""):
        self.dataset = Dataset(file_path)
        self.tree = None
        print(self.dataset.attribute_names)

    def build(self):
        self.tree = self._build_tree(self.dataset)

    def _build_tree(self, dataset):
        if len(dataset) == 0:
            raise ValueError("The dataset cannot be empty!")
        if dataset.entropy(dataset.number_of_labels()) == 0:
            return Leaf(dataset.examples[0].label)
        if len(dataset.attribute_names) <= 1:
            return Leaf(max(dataset.possible_labels(), key=lambda lbl: len(dataset.subset_by_label(lbl))))

        best_attribute = dataset.optimal_attribute()
        root = Node(best_attribute)
        for value in dataset.possible_values_for_attribute(best_attribute):
            subset = dataset.subset_by_attribute_value(best_attribute, value)
            root.children[value] = self._build_tree(subset) if subset.examples else Leaf(max(dataset.possible_labels(), key=lambda lbl: len(dataset.subset_by_label(lbl))))
        return root

    def display(self):
        self._display_tree(self.tree, 0)

    def _display_tree(self, node, indent=0):
        if isinstance(node, Node):
            print("  " * indent + f"[{node.attribute_tested}]")
            for value, child in node.children.items():
                print("  " * (indent + 1) + f"- {value}")
                self._display_tree(child, indent + 2)
        elif isinstance(node, Leaf):
            print("  " * indent + f"=> {node.label}")

    def classify(self, example):
        node = self.tree
        while isinstance(node, Node):
            attribute = node.attribute_tested
            node = node.children.get(example.attributes.get(attribute), self._most_frequent_label(node))
        example.label = node.label

    def _most_frequent_label(self, node):
        label_counts = Counter()
        def count_labels(n):
            if isinstance(n, Leaf):
                label_counts[n.label] += 1
            else:
                for child in n.children.values():
                    count_labels(child)
        count_labels(node)
        return Leaf(label_counts.most_common(1)[0][0])