import networkx as nx


class TreeNode:
    node_counter = 0

    def __init__(self, name="", values=None):
        """
        Initialize a node with a name, optional values, and an empty list of connections.
        :param name: A unique identifier for the node (e.g., 'Circle1')
        :param values: A list of values to store in this node (default: empty list)
        """
        if name == "":
            self.name = f"Node{TreeNode.node_counter}"
            TreeNode.node_counter += 1
        else:
            self.name = name

        self.values = values if values is not None else []
        self.connections = []

    def add_value(self, value):
        """Add a value to the node."""
        self.values.append(value)

    def connect_to(self, other_node):
        """
        Connect this node to another node.
        :param other_node: A TreeNode instance to connect to
        """
        if other_node not in self.connections:
            self.connections.append(other_node)

    def __repr__(self):
        return f"TreeNode({self.name}, values={self.values}, connections={[n.name for n in self.connections]})"


def build_directed_graph_from_tree(root_nodes):
    """
    Converts a TreeNode-based structure into a NetworkX directed graph.
    :param root_node: The starting TreeNode
    :return: A NetworkX directed graph object
    """
    graph = nx.DiGraph()  # Use DiGraph for directed graphs
    visited = set()

    def dfs(node):
        """Perform DFS to traverse nodes and add them to the graph."""
        if node.name in visited:
            return
        visited.add(node.name)
        graph.add_node(node.name, values=node.values)
        for connected_node in node.connections:
            graph.add_edge(node.name, connected_node.name)  # Add directed edges
            dfs(connected_node)

    for root in root_nodes:
        dfs(root)
    for root in root_nodes:
        if root.name not in graph.nodes:
            graph.add_node(root.name)
    return graph


def get_key(nested_dict, value):
    for outer_key, inner_dict in nested_dict.items():
        if "index" in inner_dict and inner_dict["index"] == value:
            return outer_key
    return None
