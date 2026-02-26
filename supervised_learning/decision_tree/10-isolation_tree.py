#!/usr/bin/env python3
"""
Module to implement an Isolation Random Tree for outlier detection
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """ Represents a random tree designed to isolate outliers """
    def __init__(self, max_depth=10, seed=0, root=None):
        """ Initializes the isolation tree """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """ Returns the string representation of the tree """
        return self.root.__str__()

    def depth(self):
        """ Returns the maximum depth of the tree """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Returns the number of nodes or leaves in the tree """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """ Updates the feature bounds for every node in the tree """
        self.root.update_bounds_below()

    def get_leaves(self):
        """ Returns a list of all leaves in the tree """
        return self.root.get_leaves_below()

    def update_predict(self):
        """ Builds the vectorized prediction function for leaf depth """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        # Predict returns the leaf.value (which is the depth)
        self.predict = lambda A: np.sum(
            [leaf.value * leaf.indicator(A) for leaf in leaves],
            axis=0
        )

    def np_extrema(self, arr):
        """ Returns the min and max of a numpy array """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """ Selects a random feature and threshold for splitting """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """ Creates a leaf child where value is the node's depth """
        # In Isolation Trees, the leaf value is its depth
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """ Creates an internal node child """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """ Recursively grows the tree to isolate points """
        node.feature, node.threshold = self.random_split_criterion(node)

        left_pop = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] > node.threshold
        )
        right_pop = np.logical_and(
            node.sub_population,
            np.logical_not(left_pop)
        )

        # Stop if max depth reached or if sub-population has only 1 individual
        is_left_leaf = (node.depth + 1 == self.max_depth or
                        np.sum(left_pop) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        is_right_leaf = (node.depth + 1 == self.max_depth or
                         np.sum(right_pop) <= self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """ Fits the isolation tree to the explanatory data """
        self.explanatory = explanatory
        # Initialize root population with all True
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"  Training finished.\n"
                  f"    - Depth                     : {self.depth()}\n"
                  f"    - Number of nodes           : {self.count_nodes()}\n"
                  f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}")
