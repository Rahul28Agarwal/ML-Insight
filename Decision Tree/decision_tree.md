# Guide to Decision Trees

_A complete walkthrough of how decision trees work with Python examples and diagrams._

Decision Tree
Created time April 29, 2025 926 AM
Last edited time May 15, 2025 459 PM
Type Algorithm
Executive Summary
Decision Trees are flexible, non-parametric models for classification and regression that 
partition data via a recursive splitting process.
They rely on impurity measures Entropy, Gini impurity) or variance reduction to choose 
splits that maximally separate labels or minimize error.
Bias–Variance Trade‑off Shallow trees underfit (high bias), deep trees overfit (high 
variance). Controlled by hyperparameters and pruning.
Pre‑pruning (e.g.,  ) and post‑pruning (cost‑complexity) prevent overfitting and 
max_depth
improve generalization.
Feature importance scores quantify each featureʼs contribution to reducing impurity.
Decision Trees underpin ensemble methods like Random Forests and Gradient Boosted 
Trees, offering improved stability and accuracy.
Table of  1. Introduction
Content
Decision Trees are supervised learners that split data into 
Executive 
homogeneous subsets by testing feature values at each node.
Summary
Table of Content Tree Structure Root node  Internal (decision) nodes  Leaf (terminal) 
Resources nodes.
1. Introduction
Advantages Interpretability, minimal preprocessing, handling of mixed 
2. Theoretical 
Foundations data types, non-parametric.
3. Splitting 
Limitations Instability (high variance), tendency to overfit without 
Criteria
control.
3.1 Entropy & 
Information 
Gain 
ID3/C4.5
3.2 Gini 
2. Theoretical Foundations
Impurity 
CART
3.3 Variance  Divide‑and‑Conquer Recursively partition feature space into 
Reduction 
axis‑aligned regions.
Regression)
4. Decision Tree  Function Approximation Piecewise-constant model: each leaf predicts 
Construction 
a constant (class label or average target).
Decision Tree 1

process Bias–Variance Trade‑off:
5. Handling 
Bias Error from erroneous model assumptions (e.g., tree too 
Continuous and 
Categorical  shallow).
Features.
Variance Error from sensitivity to training data fluctuations (e.g., 
5. Decision Tree 
tree too deep).
Algorithms
6. Pros and Cons
Optimal tree depth balances bias and variance.
Advantages
Disadvantages
3. Splitting Criteria
7. Stopping 
Criteria & 
Pruning
6.1 Stopping  3.1 Entropy & Information Gain (ID3/C4.5)
Criteria
 Entropy:
6.2 
Pre‑Pruning
Think of entropy as a measure of “surpriseˮ or “uncertaintyˮ in your 
6.3 
class distribution.
Post‑Pruning 
Cost‑Complexity)
If a node contains only one class (e.g. all “yesˮ), thereʼs zero 
8. 
surprise in drawing any sample—you always know its label—so 
Hyperparameters 
& Regularization entropy  0.
8. Practical 
If two classes are equally represented 50/50, your uncertainty is 
Example
maximal: you need 1 bit of information to decide which class you 
Classification
get.
Regression
The CART 
 
training algorithm
How the 
c
Entropy = H(S) = −∑ P log P
CART  i=1 i 2 i 
       
Algorithm 
Where: 
works
1.  P
  i  = probability of class i.
Initialization  
2.  c = classes
Splitting 
 Information Gain: 
process 
Recursive 
Measures how much entropy “dropsˮ when you split a node on a 
Partitioning)
feature AAA.
3. 
Recursion
If the split creates very pure child nodes, you get a large drop in 
4. 
entropy—and thus high Information Gain.
Stopping 
Criteria Information Gain = I(S,A) = Entropy(parent) −
5. Leaf  ∣S ∣
∑ v Entropy(S )
Nodes v∈Value(A) ∣S∣ v  
     
Decision Tree 
Where:
from Scratch
∣S∣
   The number of samples in the parent set S
Resources ∣S ∣
k   = is the number of samples that take feature value v.
 
3.2 Gini Impurity (CART)
Decision Tree 2

Gini Impurity
Gives the probability that a randomly chosen sample from the node 
would be misclassified if you label it according to the nodeʼs class 
distribution.
Like entropy, it is 0 for a pure node and higher for mixed nodes—but 
computationally simpler (no logarithms).
c
Gini = G(S) = 1 − ∑ P2
i
   
i=1
Gini Impurity = 1 − (probability of class 1)2 −
(probability of class 2)2
 
Gini impurity is 0 for pure nodes (all samples belong to one class).
3.3 Variance Reduction (Regression)
When your target y is continuous, you canʼt talk about class probabilities. 
Instead:
Mean Squared Error MSE at a node:
∣S∣
1 1
MSE(S) = ∑(y − yˉ )2,yˉ = ∑ y
i s s i
∣S∣ ∣S∣
                   
i=1 i
This is literally the variance of your labels in that node.
Variance Reduction
We pick the split that maximizes the drop in MSE
∣S ∣
v
ΔMSE = MSE(S) − ∑ MSE(S )
 
v
S
     
v
Bottom line: all these criteria are just different ways of measuring how 
“goodˮ a split is—either by reducing uncertainty (entropy/Gini) for 
classification, or by reducing prediction error (variance) for regression.
4. Decision Tree Construction process
flowchart LR
    AStart: Entire Dataset]  B{Evaluate All Possible Splits}
    B |Best Criterion| CCreate Decision Node]
    C  DStopping Criteria Met?
Decision Tree 3

    D  No  B
    D  Yes  ECreate Leaf Node]
The tree is constructed in a recursive, greedy manner, meaning it always 
chooses the best split at each step without reconsidering previous 
decisions.
 Evaluate Splits:
For each feature, compute candidate thresholds (for continuous) or 
groupings (for categorical).
At each candidate, evaluate a splitting criterion:
Classification Use Information Gain ID3, Gain Ratio C4.5, or 
Gini Impurity CART.
Regression Use variance reduction (based on Mean Squared 
Error).
The goal is to maximize the reduction in impurity.
 Select Best Split:
Choose the feature and threshold/group that results in the largest 
gain.
This split forms an internal (decision) node. Data is divided into left 
and right subsets.
 Recursive Partitioning:
Repeat the process on each child subset.
At every stage, check stopping conditions:
All samples in the subset belong to the same class.
No features remain to split on.
Node size is below  , or depth exceeds  .
min_samples_split max_depth
 Leaf Node Creation:
When a stopping condition is met, the node becomes a leaf:
Classification Assign the majority class.
Regression Assign the average of target values.
This construction strategy ensures that each path from the root to a leaf 
represents a clear, interpretable decision rule.
5. Handling Continuous and Categorical 
Features.
Continuous Features:
Decision Tree 4

These are numerical features that can take an infinite range of 
values.
The algorithm identifies possible split points by sorting all unique 
values of the feature.
It then tests thresholds placed between each pair of adjacent sorted 
values, calculating the chosen impurity metric (e.g., Gini, entropy, 
MSE.
The split that best reduces impurity becomes the decision point.
Example: For values 2, 4, 6, possible split points would be 3 and 5.
Categorical Features:
These are discrete, non-numeric values representing categories 
(e.g., color: red, green, blue).
Algorithms differ in handling:
CART: performs binary splits, creating two groups by selecting a 
subset of categories vs. the rest.
ID3 / C4.5: allow multiway splits, creating one branch per 
category.
When the number of categories is large, binary splits are preferred to 
avoid excessive branching.
Advanced implementations may group similar categories based on 
statistical similarity to improve performance.
Proper handling of feature types ensures decision trees can flexibly model 
both numerical and categorical data with minimal preprocessing.
5. Decision Tree Algorithms
Tasks
Algorithm Split Criterion Handles Tree Type
Supported
Information
ID3 Categorical Multiway Classification
Gain
Information Categorical &
C4.5 Multiway Classification
Gain Ratio Numerical
Gini Impurity
Categorical & Classification
CART Class.), MSE Binary
Numerical Regression
(Regr.)
6. Pros and Cons
Advantages
Decision Tree 5

Easy to interpret and visualize.
Handles both categorical and numerical data without scaling.
Non-parametric: no assumptions about data distribution.
Can model non-linear relationships
Disadvantages
Prone to overfitting, especially with deep trees
Sensitive to small data variations.
Can be biased towards features with more categories.
Less accurate than ensemble methods for complex tasks
7. Stopping Criteria & Pruning
6.1 Stopping Criteria
Maximum depth ( ).
max_depth
Minimum samples to split ( ).
min_samples_split
Minimum samples per leaf ( ).
min_samples_leaf
Pure node (all samples same label) or no remaining features.
6.2 Pre‑Pruning
Pre-pruning, also known as early stopping, limits the growth of the tree 
during the training process by applying specified constraints. These 
constraints help prevent overfitting by controlling model complexity.
Typical strategies include:
Limiting the maximum depth of the tree ( ).
max_depth
Requiring a minimum number of samples to perform a split 
( ).
min_samples_split
Setting a minimum number of samples required in a leaf node 
( ).
min_samples_leaf
Establishing a minimum decrease in impurity needed for a split to be 
considered ( min_impurity_decrease ).
By applying these rules, the tree construction is halted before it becomes 
overly complex, making it more generalizable.
6.3 Post‑Pruning (Cost‑Complexity)
Post-pruning involves growing the tree to its full depth and then simplifying 
it by removing parts that do not improve generalization.
Decision Tree 6

Cost-complexity pruning, used in CART, introduces a regularization 
term in the cost function:
R (T) = R(T) + α∣leaves(T)∣
α
 
where:
R(T)
  is the total misclassification error (or loss) of the tree.
α
  is a penalty term controlling the complexity.
∣leaves(T)∣
  is the number of leaf nodes in the tree.
Pruning process:
Build a fully grown tree.
Calculate the cost of removing each subtree.
Iteratively remove the subtree that gives the smallest increase in 
error, weighted by complexity.
This strategy produces a simpler, more robust model by removing branches 
that do not significantly improve prediction accuracy on a validation set.
8. Hyperparameters & Regularization
Hyperparameter Effect
max_depth Limits tree height; reduces variance.
min_samples_split Controls split necessity; avoids tiny nodes.
min_samples_leaf Ensures leaves have enough samples; smooths output.
max_features Number of features to consider per split RF.
ccp_alpha Complexity parameter for pruning CART.
8. Practical Example
Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
iris = load_iris()
clf  DecisionTreeClassifier(max_depth=2, random_state=42
clf.fit(iris.data, iris.target)
print("Feature importances:", clf.feature_importances_) 
Regression
Decision Tree 7

import numpy as np
from sklearn.tree import DecisionTreeRegressor
np.random.seed(42)
X_quad = np.random.rand(200, 1  0.5  # a single random input feature
y_quad  X_quad ** 2  0.025 * np.random.randn(200, 1
tree_reg  DecisionTreeRegressor(max_depth=2, random_state=42
tree_reg.fit(X_quad, y_quad)
This tree looks very similar to the classification tree we built earlier. The 
main difference is that instead of predicting a class in each node, it 
predicts a value. 
This prediction is the average target value of the all the training 
instances associated with this leaf node.
The CART training algorithm
Scikit-learn uses the classification and Regression Tree CART 
algorithm to train decision trees.
CART A decision tree algorithm that produces strictly binary trees for 
both classification and regression tasks.
Gini Impurity: The default impurity measure for classification in CART, 
quantifying the likelihood of incorrect classification of a randomly 
chosen element.
Variance Reduction / Mean Squared Error MSE The impurity measure
for regression in CART, aiming to minimize the variance of the target 
Decision Tree 8

variable within each node.
Binary Splits: Each split in CART divides the data into exactly two 
branches, regardless of the number of possible feature values.
Cost Function: The function minimized at each split, based on weighted 
impurity of child nodes.
How the CART Algorithm works
flowchart TD
    ARoot Node: All Data] |Best Split - min impurity| B1Left Child Node]
    A |Best Split - min impurity| B2Right Child Node]
    B1 |Continue splitting| C1Leaf or Internal Node]
    B2 |Continue splitting| C2Leaf or Internal Node]
1. Initialization
All the training data is assigned to the root node
2. Splitting process (Recursive Partitioning)
For each candidate feature and possible split point (threshold)
Classification: Compute the Gini impurity for the resulting left and 
right child nodes.
Regression: Compute the variance MSE for the resulting child 
nodes.
Calculate the cost function for the split:
Classification
⁍
Where:
m = total samples at the node
m ,m
left right  = samples in left/right child
   
G ,G
left right   Gini impurity of left/right child
   
Regression
⁍
Where:
∑(y^ −y(i))2
MSE = node
node      
m
  node  
 
∑y(i)
y^ =
node  
m
    node  
 
Decision Tree 9

3. Recursion
Partition the data into two subsets based on the chosen split.
Repeat the process for each child node (left and right), recursively 
building the tree.
4. Stopping Criteria
Stop splitting when:
All samples in a node belong to the same class (classification) or 
have very similar target values (regression).
The node contains fewer than a minimum number of samples 
(controlled by hyperparameters 
like   or  ).
min_samples_split min_samples_leaf
The maximum tree depth is reached.
No further impurity reduction is possible.
5. Leaf Nodes
Classification: Assign the most frequent class label among samples in 
the node.
Regression: Assign the mean of the target values among samples in the 
node.
Decision Tree from Scratch
Decision Tree from Scratch
Decision Tree 10

---
*Authored by Rahul Aggarwal — empowering practical machine learning, one concept at a time.*