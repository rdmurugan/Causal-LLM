# tests/test_dag_parser.py

import unittest
from causalllm.dag_parser import DAGParser

class TestDAGParser(unittest.TestCase):

    def test_topological_order(self):
        edges = [("A", "B"), ("B", "C"), ("A", "C")]
        parser = DAGParser(edges)
        order = parser.get_topological_order()
        self.assertEqual(order, ["A", "B", "C"])

    def test_invalid_dag(self):
        edges = [("A", "B"), ("B", "A")]  # cycle
        with self.assertRaises(ValueError):
            DAGParser(edges)

if __name__ == "__main__":
    unittest.main()
