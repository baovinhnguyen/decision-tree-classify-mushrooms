# decision-tree-classify-mushrooms
Use a decision tree algorithm to classify different species of mushrooms.

- Dataset: includes mushroom records drawn from the Audubon Society Field Guide to North American Mushrooms (1981). The database describes samples from different species of gilled mushrooms in the families Agaricus and Lepiota. Each sample is described by a string of 23 characters, on a single line of the provided file mushroom_data.txt; each such string describes the values of 22 attributes for each sample, and the last character corresponds to the correct classification of the mushroom into either edible (e) or poisonous (p) mushrooms.

- When the program begins, it should ask the user for three inputs:

• A training set size: this should be some integer value S that is a multiple of 250, within bounds 250 ≤ S ≤ 1000.

• A training set increment: this should be some integer value I ∈ {10,25,50} (that is, one of those three values only).

• A heuristic for use in choice of attributes when building a tree. 
