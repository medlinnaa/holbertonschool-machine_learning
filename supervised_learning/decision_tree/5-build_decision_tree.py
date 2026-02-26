#!/usr/bin/env python3
"""
a module to build a Decision Tree with indicators
"""
import numpy as np


class Node:
    """ Represents an internal node in a decision tree """
    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = None
        self.upper = None
        self.indicator = None

    def max_depth_below(self):
        """ Recursively finds the maximum depth """
        return max(
            self.left_child.max_depth_below(),
            self.right_child.max_depth_below()
        )

    def count_nodes_below(self, only_leaves=False):
        """ Recursively counts the nodes or leaves """
        left_count = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves
        )
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def left_child_add_prefix(self, text):
        """ Adds the prefix +---> for left children """
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """ Adds the prefix +---> for right children """
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """ String representation of a node """
        if self.is_root:
            out = (f"root [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")
        else:
            out = (f"node [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")

        out += self.left_child_add_prefix(self.left_child.__str__())
        out += self.right_child_add_prefix(self.right_child.__str__())
        return out

    def get_leaves_below(self):
        """ Returns a list of all leaves below this node """
        return self.left_child.get_leaves_below() + \
            self.right_child.get_leaves_below()

    def update_bounds_below(self):
        """ Recursively computes the bounds for each node """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

        self.left_child.lower[self.feature] = self.threshold
        self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """ Computes the indicator function for the node """
        def is_large_enough(x):
            """ Checks if all features are > lower bounds """
            return np.all([np.greater(x[:, key], self.lower[key])
                          for key in self.lower.keys()], axis=0)

        def is_small_enough(x):
            """ Checks if all features are <= upper bounds """
            return np.all([np.less_equal(x[:, key], self.upper[key])
                          for key in self.upper.keys()], axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                                    is_small_enough(x)]),
                                          axis=0)


class Leaf(Node):
    """ Represents a leaf node in a decision tree """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ Returns leaf depth """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """ Returns 1 for leaf count """
        return 1

    def __str__(self):
        """ String representation of a leaf """
        return f"leaf [value={self.value}]"

    def get_leaves_below(self):
        """ Returns the leaf itself in a list """
        return [self]

    def update_bounds_below(self):
        """ Leaves don't have children to update """
        pass


class Decision_Tree():
    """ Represents the decision tree itself """
    def __init__(
        self,
        max_depth=10,
        min_pop=1,
        seed=0,
        split_criterion="random",
        root=None
    ):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """ Returns tree depth """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Returns tree node count """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """ Returns the full tree string """
        return self.root.__str__()

    def get_leaves(self):
        """ Returns a list of all leaves in the tree """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """ Starts the bounds update from the root """
        self.root.update_bounds_below()
