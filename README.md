# ID3 Decision Tree Classifier

This repository contains a Python implementation of the **ID3 algorithm** for building a decision tree. The code reads a CSV dataset, constructs a decision tree using information gain, and uses it to classify new examples.

---

## Features

- Implements the ID3 algorithm from scratch.
- Reads/processes CSV datasets with categorical features.
- Classifies new examples using the trained tree.
- graphical user interface (GUI) for dataset upload, tree visualization, and predictions.

---

## Requirements

- Python 3.7+

---

## Usage


### Example Output

---

## Dataset Details

The included dataset (`data.csv`) contains 14 entries about weather conditions and tennis playability:

| Attribute       | Values                          |
|-----------------|---------------------------------|
| Prévisions       | Ensoleillé, Nuageux, Pluvieux  |
| Température     | Chaud, Frais, Moyen            |
| Humidité        | Élevée, Normale                |
| Vent            | Faible, Fort                   |
| Classe (Target) | Oui (Yes), Non (No)            |

---

## file Structure

- **`Dataset`**: Manages data loading and processing.
- **`Node`**: Tree structure components.
- **`ID3Tree`**: Implements the ID3 algorithm and classification.

---





