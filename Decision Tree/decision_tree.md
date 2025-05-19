# Guide to Decision Trees

_A complete walkthrough of how decision trees work with Python examples and diagrams._

## 📚 Introduction
Decision trees are one of the most intuitive and interpretable machine learning models. They work by recursively splitting data based on feature values, forming a tree-like structure that helps in both classification and regression tasks.

## 🔍 Why Decision Trees Matter
Decision trees:
- Require minimal data preparation
- Handle both numerical and categorical features
- Are easy to visualize and explain to stakeholders
- Can form the basis for powerful ensemble methods like Random Forests and Gradient Boosted Trees

## 🧠 How Decision Trees Work
The algorithm starts at the root and selects the feature that best splits the data according to a certain criterion (like Information Gain or Gini Index). It continues splitting until:
- All samples belong to one class
- No features are left to split
- A maximum depth or minimum samples condition is met

## 🧩 Key Concepts

### Entropy
A measure of disorder or uncertainty in a dataset:

\[Entropy(S) = - \sum p_i \log_2(p_i)\]

### Information Gain
The decrease in entropy after a dataset is split on an attribute:

\[IG(S, A) = Entropy(S) - \sum \left(\frac{|S_v|}{|S|}\right) Entropy(S_v)\]

### Gini Index
An impurity measure used in the CART algorithm:

\[Gini(S) = 1 - \sum p_i^2\]

### CART Algorithm
- Uses the Gini index
- Performs binary splits
- Applicable to both classification and regression

### ID3 Algorithm
- Uses entropy and information gain
- Supports multi-way splits (not just binary)

### Pruning
To prevent overfitting:
- **Pre-pruning**: Limits tree depth or min samples
- **Post-pruning**: Trims branches from a fully grown tree

## 🌳 Types of Decision Trees
| Algorithm | Criterion       | Splitting     | Used For     |
|-----------|-----------------|---------------|--------------|
| ID3       | Information Gain| Multi-way     | Classification |
| CART      | Gini Index      | Binary        | Classification & Regression |

## 💻 Code Examples with Visualization

### Decision Tree Classifier
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

iris = load_iris()
clf = DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

plt.figure(figsize=(10,6))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```

### Decision Tree Regressor
```python
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

housing = fetch_california_housing()
reg = DecisionTreeRegressor()
reg.fit(housing.data, housing.target)

plt.figure(figsize=(10,6))
plot_tree(reg, feature_names=housing.feature_names, filled=True)
plt.show()
```

## ✅ Advantages
- Easy to interpret and visualize
- Requires little preprocessing
- Handles both numerical and categorical variables

## ⚠️ Disadvantages
- Prone to overfitting
- Can be unstable with small variations in data
- Greedy approach may not find global optimum

## 📈 Real-World Applications
- **Medical Diagnosis**: Predicting diseases based on symptoms
- **Credit Scoring**: Evaluating loan eligibility
- **Customer Segmentation**: Grouping users for targeted marketing


*Authored by Rahul Aggarwal — empowering practical machine learning, one concept at a time.*