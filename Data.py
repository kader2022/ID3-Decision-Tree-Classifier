from collections import Counter
from math import log

class Example:
    def __init__(self, attribute_names, attribute_values, label=""):
        if len(attribute_values) != len(attribute_names):
            raise ValueError("attribute_names and attribute_values must have the same length")
        self.label = label
        self.attributes = dict(zip(attribute_names, attribute_values))

class Dataset:
    def __init__(self, df=None):
        self.attribute_names = []
        self.examples = []
        
        if df is not None:
            self._load_from_dataframe(df)
    
    def _load_from_dataframe(self, df):
        """Load data from pandas DataFrame"""
        # Get column names
        self.attribute_names = list(df.columns)
        
        # Create examples
        for _, row in df.iterrows():
            values = list(row.values)
            self.examples.append(Example(
                self.attribute_names[:-1], 
                values[:-1], 
                str(values[-1])
            ))
    
    def __len__(self):
        return len(self.examples)
    
    def possible_labels(self):
        return list(set(example.label for example in self.examples))
    
    def number_of_labels(self):
        return len(self.possible_labels())
    
    def subset_by_label(self, label):
        return Dataset.from_examples(self.attribute_names, [e for e in self.examples if e.label == label])
    
    def entropy(self, num_of_labels):
        if len(self.examples) == 0:
            return 0
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
        
        weighted_entropy = 0
        for value in self.possible_values_for_attribute(attribute_name):
            subset = self.subset_by_attribute_value(attribute_name, value)
            if len(subset) > 0:
                weighted_entropy += (len(subset) / total_examples) * subset.entropy(self.number_of_labels())
        
        return total_entropy - weighted_entropy
    
    def optimal_attribute(self):
        if len(self.attribute_names) <= 1:
            return None
        return max(self.attribute_names[:-1], key=self.information_gain)
    
    @classmethod
    def from_examples(cls, attribute_names, examples):
        dataset = cls()
        dataset.attribute_names = attribute_names[:]
        dataset.examples = examples
        return dataset


