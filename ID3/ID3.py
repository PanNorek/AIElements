import math
from collections import deque

class Node:

    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None


class DecisionTreeClassifier:

    def __init__(self, X, feature_names, labels):
        self.X = X
        self.feature_names = feature_names
        self.labels = labels
        self.labelCategories = list(set(labels))
        self.labelCategoriesCount = [list(labels).count(x) for x in self.labelCategories]
        self.node = None
        self.entropy = self._get_entropy([x for x in range(len(self.labels))])

    def _get_entropy(self, x_ids):

        labels = [self.labels[i] for i in x_ids]
        label_count = [labels.count(x) for x in self.labelCategories]
        entropy = sum([-count / len(x_ids) * math.log(count / len(x_ids), 2) if count else 0 for count in label_count])
        return entropy

    def _get_information_gain(self, x_ids, feature_id):
        info_gain = self._get_entropy(x_ids)
        x_features = [self.X[x][feature_id] for x in x_ids]
        feature_vals = list(set(x_features))
        feature_vals_count = [x_features.count(x) for x in feature_vals]
        feature_vals_id = [
            [x_ids[i]
            for i, x in enumerate(x_features)
            if x == y]
            for y in feature_vals
        ]

        info_gain = info_gain - sum([val_counts / len(x_ids) * self._get_entropy(val_ids)
                                     for val_counts, val_ids in zip(feature_vals_count, feature_vals_id)])

        return info_gain

    def _get_feature_max_information_gain(self, x_ids, feature_ids):

        features_entropy = [self._get_information_gain(x_ids, feature_id) for feature_id in feature_ids]
        max_id = feature_ids[features_entropy.index(max(features_entropy))]

        return self.feature_names[max_id], max_id

    def id3(self):

        x_ids = [x for x in range(len(self.X))]
        feature_ids = [x for x in range(len(self.feature_names))]
        self.node = self._id3_recv(x_ids, feature_ids, self.node)
        print('')

    def _id3_recv(self, x_ids, feature_ids, node):

        if not node:
            node = Node()
        labels_in_features = [self.labels[x] for x in x_ids]
        if len(set(labels_in_features)) == 1:
            node.value = self.labels[x_ids[0]]
            return node
        if len(feature_ids) == 0:
            node.value = max(set(labels_in_features), key=labels_in_features.count)
            return node
        best_feature_name, best_feature_id = self._get_feature_max_information_gain(x_ids, feature_ids)
        node.value = best_feature_name
        node.childs = []
        feature_values = list(set([self.X[x][best_feature_id] for x in x_ids]))
        for value in feature_values:
            child = Node()
            child.value = value
            node.childs.append(child)
            child_x_ids = [x for x in x_ids if self.X[x][best_feature_id] == value]
            if not child_x_ids:
                child.next = max(set(labels_in_features), key=labels_in_features.count)
                print('')
            else:
                if feature_ids and best_feature_id in feature_ids:
                    to_remove = feature_ids.index(best_feature_id)
                    feature_ids.pop(to_remove)
                child.next = self._id3_recv(child_x_ids, feature_ids, child.next)
        return node

    def printTree(self):
        if not self.node:
            return
        nodes = deque()
        nodes.append(self.node)
        while len(nodes) > 0:
            node = nodes.popleft()
            print(node.value)
            if node.childs:
                for child in node.childs:
                    print('({})'.format(child.value))
                    nodes.append(child.next)
            elif node.next:
                print(node.next)
