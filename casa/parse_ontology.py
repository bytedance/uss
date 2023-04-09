'''
import json
from casa.config import IDS, ID_TO_IX


class Node:
    def __init__(self, data):
        r"""

        Args:
            data: dict, e.g., {
                'id': '/m/0dgw9r', 
                'name': 'Human sounds', 
                'description': ..., 
                child_ids: [...],
                ...}
        """
        
        self.id = data['id']
        self.data = data
        self.children = []

    def search(self, id):
        nodes, layers = self.traverse(self)

        for node in nodes:
            if id == node.id:
                return node

        return None

    def search_parent(self, id):
        nodes, layers = self.traverse(self)

        for node in nodes:
            if id in node.data['child_ids']:
                return node
        return None

    def traverse(self, node, layer=0):
        nodes = [node]
        layers = [layer]
        for e in node.children:
            _nodes, _layers = self.traverse(e, layer + 1)
            nodes += _nodes
            layers += _layers

        return nodes, layers



def get_ontology_tree(verbose=True):

    ontology_path = './metadata/ontology.json'

    with open(ontology_path) as f:
        data_list = json.load(f)    # len: 632

    root = Node(data={'id': '0', 'name': '0', 'child_ids': []})

    for data in data_list:
        # E.g., data: {'id': '/m/0dgw9r', 'name': 'Human sounds', 'description': ..., child_ids: [...]}

        id = data['id']
        node = root.search_parent(id)

        if not node:
            root.children.append(Node(data))
        else:
            node.children.append(Node(data))

    if not verbose:
        return root

    else:

        nodes, layers = root.traverse(root)
        
        for node, layer in zip(nodes, layers):
            print('  '*layer, layer, node.data['name'], node.data['id'])

        ###
        node = root.search(id='/m/0dgw9r')
        nodes, layers = root.traverse(node)

        id_list = []
        for node in nodes:
            id_list.append(node.data['id'])

        # print(id_list)

        index_list = []
        for id in id_list:
            if id in IDS:
                index_list.append(ID_TO_IX[id])
        # print(index_list)
        return root

        # for node, layer in zip(nodes, layers):
        #     print('  '*layer, layer, node.data['name'], node.data['id'])


def get_subclass_indexes(root, id):
    r"""Get indexes of all sub tree sound classes.

    Args:
        root: Node
        id: str

    Returns:
        index_list: list of int
    """

    node = root.search(id=id)
    nodes, layers = root.traverse(node)

    id_list = []
    for node in nodes:
        id_list.append(node.data['id'])

    index_list = []
    for id in id_list:
        if id in IDS:
            index_list.append(ID_TO_IX[id])
            
    return index_list


if __name__ == '__main__':
    get_ontology_tree()
'''

import json
from casa.config import IDS, ID_TO_IX


class Node:
    def __init__(self, data, depth):
        r"""

        Args:
            data: dict, e.g., {
                'id': '/m/0dgw9r', 
                'name': 'Human sounds', 
                'description': ..., 
                child_ids: [...],
                ...}
        """
        
        self.class_id = data['id']
        self.data = data
        self.children = []
        self.depth = depth

    @staticmethod
    def search(node, class_id):

        if node.class_id == class_id:
            return node

        else:
            for child in node.children:
                result = Node.search(node=child, class_id=class_id)
                if result:
                    return result

        return None

    @staticmethod
    def search_parent(node, class_id):

        if class_id in node.data['child_ids']:
            return node

        else:
            for child in node.children:
                result = Node.search_parent(node=child, class_id=class_id)
                if result:
                    return result

        return None

    @staticmethod
    def traverse(node):

        nodes = [node]

        for child in node.children:
            nodes.extend(Node.traverse(node=child))

        return nodes

root_class_id_dict = {
    '/m/0dgw9r': "Human sounds",
    '/m/0jbk': "Animal",
    '/m/04rlf': "Music",
    '/m/059j3w': "Natural sounds",
    '/t/dd00041': "Sounds of things",
    '/t/dd00098': "Source-ambiguous sounds",
    '/t/dd00123': "Channel, environment and background",
}



def get_ontology_tree(verbose=True):

    ontology_path = './metadata/ontology.json'

    with open(ontology_path) as f:
        data_list = json.load(f)    # len: 632

    root_class_ids = list(root_class_id_dict.keys())

    data = {
        'id': 'root',
        'name': 'root',
        'child_ids': root_class_ids,
    }

    root = Node(data=data, depth=0)

    for data in data_list:
        # E.g., data: {'id': '/m/0dgw9r', 'name': 'Human sounds', 'description': ..., child_ids: [...]}

        # print(data)

        father = Node.search_parent(node=root, class_id=data['id'])

        child = Node(data=data, depth=father.depth + 1)
        
        father.children.append(child)

    if verbose:

        nodes = Node.traverse(node=root)

        for node in nodes:
            print(node.depth, node.data['name'], node.data['id'])
        
        ###
        node = root.search(node=root, class_id='/m/0dgw9r')
        
        nodes = root.traverse(node=node)

    return root


def get_subclass_indexes(root, id):
    r"""Get indexes of all sub tree sound classes.

    Args:
        root: Node
        id: str

    Returns:
        index_list: list of int
    """

    node = root.search(id=id)
    nodes, layers = root.traverse(node)

    id_list = []
    for node in nodes:
        id_list.append(node.data['id'])

    index_list = []
    for id in id_list:
        if id in IDS:
            index_list.append(ID_TO_IX[id])
            
    return index_list


if __name__ == '__main__':
    get_ontology_tree()