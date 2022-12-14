import math

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# Test dataset for classification
import seaborn as sns
import matplotlib.pyplot as plt

# Trying to not use pandas as it'd probably be better for learning
class DecisionTree:
    def __init__(self, x, y):
        self.X = x
        self.Y = y

    def create_tree(self):
        left, right, col = self.calculate_root_split()

    def calculate_root_split(self):
        gini_values = []
        left_split = []
        right_split = []

        X = self.X.transpose()
        for idx, col in enumerate(X):
            gini_values.append((idx, DecisionTree.calc_gini_col(col, self.Y)))

        # Split on lowest impurity
        lowest_gini_impurity = sorted(gini_values, key=lambda x: x[1][1])[0]
        col_to_split = lowest_gini_impurity[0]
        split_index = lowest_gini_impurity[1][0]

        value_to_split = X[col_to_split][split_index]
        # print(lowest_gini_impurity)

        # You already made this function, maybe try use it
        for idx, i in enumerate(X[col_to_split]):
            if i < value_to_split:
                temp = [j[idx] for j_index, j in enumerate(X) if j_index != col_to_split]
                left_split.append(temp)
            else:
                temp = [j[idx] for j_index, j in enumerate(X) if j_index != col_to_split]
                right_split.append(temp)
        print(left_split, right_split)
        return left_split, right_split

    @staticmethod
    def calc_gini(x):
        impurity = 0
        for i in x:
            impurity += (i/sum(x))**2
        return 1 - impurity

    @staticmethod
    def calc_gini_col(col: list, target: list):
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
    def create_split(col, target, val):
        left_split = []
        right_split = []
        # For each value in the column, we have to get their individual gini impurity
        # Create split on average age

        for idx, i in enumerate(col):
            if i < val:
                left_split.append(target[idx])
            else:
                # Take sum of classifications in right split
                # count(0) in right split or something
                right_split.append(target[idx])
        return left_split, right_split

def main():
    data = load_iris()
    tree = DecisionTree(data.data, data.target)
    print(len(tree.X), len(tree.Y))
    # print(tree.X)
    tree.calculate_root_split()

    # train_data, test_data = train_test_split(data, test_size=0.33)
    # train_df = pd.DataFrame(train_data, columns=data.feature_names)
    # # print(type(df))
    # print(train_df.head)
    # print(train_df.columns)
    # sns.heatmap(train_df.corr(), annot=True)
    # plt.show()


if __name__ == "__main__":
    main()
