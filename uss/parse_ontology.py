import json
from typing import List

from uss.config import ID_TO_IX, IDS, ROOT_CLASS_ID_DICT


class Node:
    def __init__(self, data, level):
        r"""Sound class Node.

        Args:
            data: dict, e.g., {
                "id": "/m/0dgw9r",
                "name": "Human sounds",
                "description": ...,
                child_ids: [...],
                ...}
        """

        self.class_id = data["id"]
        self.data = data
        self.children = []
        self.level = level

    @staticmethod
    def search(node, class_id: str):  # -> Union[Node, None]:
        r"""Search the node with class_id in the ontology tree."""

        if node.class_id == class_id:
            return node

        else:
            for child in node.children:
                result = Node.search(node=child, class_id=class_id)
                if result:
                    return result

        return None

    @staticmethod
    def search_parent(node, class_id: str):  # -> Union[Node, None]:
        r"""Search the parent of a node with class_id."""

        if class_id in node.data["child_ids"]:
            return node

        else:
            for child in node.children:
                result = Node.search_parent(node=child, class_id=class_id)
                if result:
                    return result

        return None

    @staticmethod
    def traverse(node):  # -> List[Node]:
        r"""Traver all children of a Node including itself."""

        nodes = [node]

        for child in node.children:
            nodes.extend(Node.traverse(node=child))

        return nodes


def get_ontology_tree(ontology_path: str) -> Node:
    r"""Parse and build the AudioSet ontology tree."""

    with open(ontology_path) as f:
        data_list = json.load(f)    # len: 632

    root_class_ids = list(ROOT_CLASS_ID_DICT.keys())

    data = {
        "id": "root",
        "name": "root",
        "child_ids": root_class_ids,
    }

    root = Node(data=data, level=0)

    for data in data_list:
        # E.g., data: {"id": "/m/0dgw9r", "name": "Human sounds",
        # "description": ..., child_ids: [...]}

        father = Node.search_parent(node=root, class_id=data["id"])

        child = Node(data=data, level=father.level + 1)

        father.children.append(child)

    return root


def get_subclass_indexes(root: Node, id: str) -> List[int]:
    r"""Get class indexes of all children of a node with id=id.

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
        id_list.append(node.data["id"])

    index_list = []
    for id in id_list:
        if id in IDS:
            index_list.append(ID_TO_IX[id])

    return index_list


if __name__ == "__main__":

    get_ontology_tree(ontology_path="./metadata/ontology.json")
