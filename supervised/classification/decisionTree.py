import math

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from itertools import groupby

import pandas as pd
import numpy as np
# Test dataset for classification
import seaborn as sns
import matplotlib.pyplot as plt

# Trying to not use pandas as it'd probably be better for learning

class DecisionTree:
    def __init__(self, x, y, depth=3):
        self.X = x.transpose()
        self.Y = y
        self.depth = depth
        self.depth_counter = 0
        self.tree_order = []

    # ----- Splitting criteria -----
    # Gini impurity
    @staticmethod
    def calc_gini(x):
        impurity = 0
        for i in x:
            impurity += (i/sum(x))**2
        return 1 - impurity

    # Entropy
    @staticmethod
    def calc_entropy(x):
        entropy = 0
        for i in x:
            entropy += (i/sum(x)) * math.log2(i/sum(x))
        return -entropy

    # Information gain
    @staticmethod
    def calc_info_gain(x):
        return 0 # TODO
    # -----------------

    @staticmethod
    def get_new_x_y(x):
        print(f"X: {x}, length: {len(x)}")
        new_X = np.array([i[:-1] for i in x])
        new_Y = np.array([i[-1] for i in x])
        print(f"newx newy {len(new_X)}, {len(new_Y)}")
        if len(new_Y) == 2:
            return new_X, new_Y[0]
        else:
            return new_X, new_Y

    @staticmethod
    def create_split(col, target, val):
        left_split = []
        right_split = []
        # For each value in the column, we have to get their individual gini impurity
        # Create split on average age

        # FIXED - LAZY FIX
        for idx, i in enumerate(col):
            if i < val:
                # if len(target) == 1:
                #     left_split.append(target[0][idx])
                # else:
                left_split.append(target[idx])
            else:
                # Take sum of classifications in right split
                # count(0) in right split or something
                # if len(target) == 1:
                #     right_split.append(target[0][idx])
                # else:
                right_split.append(target[idx])
        return left_split, right_split

    # Refactor for different criteria
    @staticmethod
    def calc_split_for_col(col: list, target: list, criteria: str = 'gini'):
        highest_gini = [0, 0]
        for idx in range(len(col)-1):
            avg = (col[idx] + col[idx+1])/2
            left_split, right_split = DecisionTree.create_split(col, target, avg)

            # Calculate gini for left and right split
            # Left Split
            # For each unique value, count the number of times it appears
            # Then calculate gini value from this
            _, count = np.unique(left_split, return_counts=True)
            gini_left = DecisionTree.calc_gini(count)

            _, count = np.unique(right_split, return_counts=True)
            gini_right = DecisionTree.calc_gini(count)

            total_gini = (len(left_split)/len(col)) * gini_left + (len(right_split)/len(col)) * gini_right
            if total_gini > highest_gini[1]:
                highest_gini[0] = idx
                highest_gini[1] = total_gini
        return highest_gini

    @staticmethod
    def calculate_root_split(X, Y):
        gini_values = []
        left_split = []
        right_split = []

        for idx, col in enumerate(X):
            gini_values.append((idx, DecisionTree.calc_split_for_col(col, Y)))

        # Split on lowest impurity
        lowest_gini_impurity = sorted(gini_values, key=lambda x: x[1][1])[0]
        col_to_split = lowest_gini_impurity[0]
        split_index = lowest_gini_impurity[1][0]

        value_to_split = X[col_to_split][split_index]

        # You already made this function, maybe try use it: create_split
        for idx, i in enumerate(X[col_to_split]):
            if i < value_to_split:
                temp = [j[idx] for j_index, j in enumerate(X) if j_index != col_to_split]
                temp.append(Y[idx])
                left_split.append(temp)
            else:
                temp = [j[idx] for j_index, j in enumerate(X) if j_index != col_to_split]
                temp.append(Y[idx])
                right_split.append(temp)
        return left_split, right_split, value_to_split, col_to_split

    def create_tree(self, params=None):
        print(self.depth_counter)
        if params is None:
            params = (self.X, self.Y)
        while self.depth_counter < self.depth:
            self.depth_counter += 1
            left, right, value, col = DecisionTree.calculate_root_split(params[0], params[1])
            self.tree_order.append((col, value))

            # Needs fixing for recursion, however nearly done
            if all(i[-1] == left[0][-1] for i in left):
                new_right_x, new_right_y = DecisionTree.get_new_x_y(right)
                right_tree = DecisionTree.calculate_root_split(new_right_x.transpose(), new_right_y.transpose())
                print(f"left: {right_tree[0]} \n right: {right_tree[1]} \n value: {right_tree[2]} \n col: {right_tree[3]}")
                return self.create_tree((right_tree[0], right_tree[1]))
            elif all(i[-1] == right[0][-1] for i in right):
                return
            else:
                new_left_x, new_left_y = DecisionTree.get_new_x_y(left)
                left_tree = DecisionTree.calculate_root_split(new_left_x.transpose(), new_left_y.transpose())
                print(f"left: {left_tree[0]} \n right: {left_tree[1]} \n value: {left_tree[2]} \n col: {left_tree[3]}")
                return self.create_tree((left_tree[0], left_tree[1]))
        return self.tree_order

                # left_tree = DecisionTree(new_left_x, new_left_y)
            # print(m)
            # new_left, new_right, new_value, new_col = DecisionTree.calculate_root_split(new_left_x, new_left_y)
            # print(new_left, new_right)
            # print(left, right, value, self.X[col])
            # split_order_dict[col] = value
            # print(split_order_dict)
def main():
    data = load_iris()
    tree = DecisionTree(data.data, data.target)
    print(len(tree.X), len(tree.Y))
    # print(tree.X)
    # tree.calculate_root_split()
    print(tree.create_tree())
    print(tree.tree_order)

    # train_data, test_data = train_test_split(data, test_size=0.33)
    # train_df = pd.DataFrame(train_data, columns=data.feature_names)
    # # print(type(df))
    # print(train_df.head)
    # print(train_df.columns)
    # sns.heatmap(train_df.corr(), annot=True)
    # plt.show()


if __name__ == "__main__":
    main()
