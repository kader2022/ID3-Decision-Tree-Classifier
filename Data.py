from collections import Counter
from math import log
import csv

class Example:
    def __init__(self, attribute_names, attribute_values, label=""):
        if len(attribute_values) != len(attribute_names):
            raise ValueError("attribute_names and attribute_values must have the same length")
        self.label = label
        self.attributes = dict(zip(attribute_names, attribute_values))

class Dataset:

    def __init__(self, file_path=""):
        self.attribute_names = []
        self.examples = []
        if file_path:
            self._load_data(file_path)

    def _load_data(self, file_path):
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                rows = [row for row in reader if row]

                if not rows:
                    raise ValueError("The file is empty or does not contain valid data.")

                self.attribute_names = [attr.lower() for attr in rows[0]]

                self.examples = [self._parse_example(row) for row in rows[1:] if len(row) == len(self.attribute_names)]

        except FileNotFoundError:
            raise ValueError(f"The file '{file_path}' does not exist!")

    def _parse_example(self, row):
        return Example(self.attribute_names[:-1], row[:-1], row[-1])

    def __len__(self):
        return len(self.examples)

    def possible_labels(self):
        return list(set(example.label for example in self.examples))

    def number_of_labels(self):
        return len(self.possible_labels())

    def subset_by_label(self, label):
        return Dataset.from_examples(self.attribute_names, [e for e in self.examples if e.label == label])

    def entropy(self, num_of_labels):
        label_counts = Counter(e.label for e in self.examples)
        total = len(self.examples)
        return -sum((count / total) * log(count / total, 2) for count in label_counts.values())

    def possible_values_for_attribute(self, attribute_name):
        return list(set(e.attributes.get(attribute_name) for e in self.examples))

    def subset_by_attribute_value(self, attribute_name, value):
        return Dataset.from_examples(self.attribute_names, [e for e in self.examples if e.attributes.get(attribute_name) == value])

    def information_gain(self, attribute_name):
        total_entropy = self.entropy(self.number_of_labels())
        total_examples = len(self)
        return total_entropy - sum((len(subset) / total_examples) * subset.entropy(self.number_of_labels()) for value in self.possible_values_for_attribute(attribute_name) if (subset := self.subset_by_attribute_value(attribute_name, value)))

    def optimal_attribute(self):
        return max(self.attribute_names[:-1], key=self.information_gain)

    @classmethod
    def from_examples(cls, attribute_names, examples):
        dataset = cls()
        dataset.attribute_names = attribute_names[:]
        dataset.examples = examples
        return dataset
