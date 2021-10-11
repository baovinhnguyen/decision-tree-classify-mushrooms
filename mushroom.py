import sys
import os
import random
import math

# Load data
#path = os.getcwd() + "/mushroom/hw01_01/input_files/mushroom_data.txt"
#input_file = open(path, "r")
#os.chdir(os.path.dirname(sys.argv[0]))
input_file = open("mushroom_data.txt", "r")
x = input_file.readlines()
input_file.close()
data = [line.split() for line in x]

class DecisionTree():
    """
    This class represents a decision tree.
    """
    branch_num = 0

    def __init__(self, root_feature = -1, guess = None):
        self.root_feature = root_feature  # root_feature = -1: leaf
        self.branches = {}  # list of branches
        self.guess = guess

    def add_branch(self, label, subtree):
        self.branches[label] = subtree

    def print_tree(self, text, out):
        if text == "":
            self.branch_num = 0
        if self.root_feature == -1:
            if self.guess == 'p':
                text += " Poison."
            else:
                text += " Edible."
            out.write(f"Branch [{self.branch_num}]:" + text + "\n")
            self.branch_num += 1
            return None
        text = text + f' Attrib #{self.root_feature}: '
        for branch in self.branches:
            text2 = text + f'{branch[0]};'
            self.branches[branch].print_tree(text2, out)
        return None

    def predict(self, obs):
        if self.root_feature == -1:
            return self.guess
        else:
            feature = self.root_feature
            value = obs[feature]
            subtree = self.branches[value]
            return subtree.predict(obs)

    def accuracy(self, test_data, out):
        no_correct = sum([(self.predict(obs) == obs[-1])*1 for obs in test_data])
        no_cases = len(test_data)
        accuracy = no_correct / no_cases
        out.write(f'Given current tree, there are {no_correct} correct classifications out of {no_cases} possible (a success rate of {accuracy:.2%} percent).\n')
        return accuracy



def DecisionTreeTrain(remaining_features, data, parent_guess, heuristic):

    # Base cases
    if len(data) == 0:
        return DecisionTree(root_feature = -1, guess = parent_guess)

    # case 2: all remaining observations have same outcome (e.g. all 'poison')
    # then cannot divide further, assign it as a leaf with the corresponding guess
    outcomes_set = set([obs[-1] for obs in data])
    if len(outcomes_set) == 1:
        return DecisionTree(-1, list(outcomes_set)[0])

    # case 3: no more remaining features to divide --> return most likely outcome
    guess = 0  # most likely outcome
    max_cnt = -1  # how often does most likely outcome appear
    outcomes =[obs[-1] for obs in data]
    for outcome in outcomes_set:
        cnt = outcomes.count(outcome)  # how many outcomes have this value
        if cnt > max_cnt:
            max_cnt = cnt
            guess = outcome
    if len(remaining_features) == 0:
        return DecisionTree(root_feature = -1, guess = guess)

    # Recursive
    # choose best feature to divide data
    best_feature = MostImportantFeature(remaining_features, data, heuristic)
    # remove that feature from the list of remaining features
    remaining_features.remove(best_feature)

    # create a tree with that "best feature" as a root
    Tree = DecisionTree(root_feature = best_feature)
    # retrieve set of values within that best feature
    value_set = features[best_feature]
    for value in value_set:
        # divide data
        subdata = [line for line in data if line[best_feature] == value]
        # train subtree
        subtree = DecisionTreeTrain(remaining_features, subdata, guess, heuristic)
        # add this branch
        Tree.add_branch(value, subtree)

    return Tree

def MostImportantCounting(remaining_features, data):
    current_max_cnt = -1
    current_max_feature = 0

    for feature in remaining_features:  # consider all remaining features, for ex. feature = "color"
        value_set = set([obs[feature] for obs in data])
        total_correct = 0
        for value in value_set: # iterate over set of possible values for colors, e.g. {"red", "blue", "green"}
            x = [obs for obs in data if obs[feature] == value]  # get subsample of mushroom w/ color = "red"
            N_x = len([obs for obs in x if obs[-1] == 'e']) # count no. of cases that is not poison
            total_correct += max(N_x, len(x) - N_x)
        if total_correct > current_max_cnt:
            current_max_cnt = total_correct
            current_max_feature = feature

    return current_max_feature


def MostImportantInfo(remaining_features, data):
    N = len(data)  # total no. of cases
    current_min_entropy = float("inf")
    current_best_feature = remaining_features[0]

    def calculate_entropy(prob): # Given a vector of probabilities, calculate entropy
        # remove the elements with zero probability
        prob = [i for i in prob if i != 0]
        return -sum([i*math.log(i, 2) for i in prob])

    for feature in remaining_features:  # search over remaining features
        remaining_entropy = 0
        value_set = set([obs[feature] for obs in data])
        for value in value_set:  # for a given feature, calculate entropy of each branch
            x = [obs for obs in data if obs[feature] == value]
            N_x = len(x)  # no. of cases with this value
            N_x_e = len([obs for obs in x if obs[-1] == "e"])  # no. of cases without poison
            entropy = calculate_entropy([N_x_e/N_x, 1 - N_x_e/N_x])
            remaining_entropy += N_x/N*entropy
        if current_min_entropy > remaining_entropy:
            current_min_entropy = remaining_entropy
            current_best_feature = feature

    return current_best_feature

def MostImportantFeature(remaining_features, data, heuristic):
    if heuristic == 'C':
        return MostImportantCounting(remaining_features, data)
    else:
        return MostImportantInfo(remaining_features, data)



# Ask user to input training parameters
S = input("Input your training set size(either 250, 500, 750, or 1000): ")
try:
    if int(S) not in [250, 500, 750, 1000]:
        print("You did not enter a valid value for traing set size.")
except ValueError:
    print("You did not enter an integer.")
S = int(S)
I = input("Please enter training increment (either 10, 25, or 50): ")
try:
    if int(I) not in [10, 25, 50]:
        print("You did not enter a valid value for traing set increment.")
except ValueError:
    print("You did not enter an integer.")
I = int(I)
H = input("Please enter heuristic for choosing attributes ([C]ounting or [I]nformation):")
if H not in ['C', 'I']:
    print("You did not enter a valid value for traing set size.")



# Create file to write output
output_name = "result" + f'_S={S}_I={I}_H={H}'+'.txt'
out = open(output_name, 'w+')
out.write(f"Training mushroom with training size S={S}, increment I={I}, and using heuristic {H}.\n\n")
# seperate data into training set (S observations) and test set
training_set_ind = random.sample(range(len(data)), k=S)
training_set = [data[i] for i in training_set_ind]
test_set = [data[i] for i in range(len(data)) if i not in training_set_ind]

# train data on subsets of training set
features = {feature_no: set([obs[feature_no] for obs in data]) for feature_no in list(range(len(data[0])-1))}
features.keys()

train_results = {}
for subset_size in range(I, S+1, I):
    training_subset = random.sample(training_set, subset_size)
    out.write(f'Running with {subset_size} examples in training set.\n')
    x = DecisionTreeTrain(list(features.keys()), training_subset, 0, H)
    train_results[subset_size] = x.accuracy(test_set, out)

# print tree to output
x.print_tree(text = "", out=out)

# close output file
out.close()

# Plot training accuracy graph
import matplotlib.pyplot as plt
lists = sorted(train_results.items())
x, y = zip(*lists)
plt.plot(x, y)
plot_name = f"accuracy_S={S}_I={I}_H={H}.png"
plt.savefig(plot_name)
