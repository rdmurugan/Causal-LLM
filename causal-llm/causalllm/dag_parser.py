# causalllm/dag_parser.py

import networkx as nx

class DAGParser:
    def __init__(self, edges):
        """
        Initialize with list of edges: e.g., [("A", "B"), ("B", "C")]
        """
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edges)

        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Input graph must be a DAG")

    def get_topological_order(self):
        """Return nodes in topological (causal) order."""
        return list(nx.topological_sort(self.graph))

    def to_prompt(self, task_description=""):
        """
        Convert DAG into a chain-of-thought prompt for LLMs.
        """
        ordered_nodes = self.get_topological_order()
        reasoning = "\n".join([f"Step {i+1}: Consider '{node}'." for i, node in enumerate(ordered_nodes)])
        prompt = f"{task_description.strip()}\n\nCausal Reasoning Steps:\n{reasoning}"
        return prompt

    def visualize(self, path="dag.png"):
        """Optional: Save a visualization of the DAG"""
        import matplotlib.pyplot as plt
        nx.draw(self.graph, with_labels=True, node_color='lightblue', arrows=True)
        plt.savefig(path)
        plt.close()
