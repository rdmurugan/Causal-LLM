# causalllm/scm_explainer.py

from typing import List, Tuple
import re


class SCMExplainer:
    """
    Extracts structural causal models (SCMs) from natural language descriptions.
    """

    def __init__(self, llm_client=None, model="gpt-4"):
        self.llm_client = llm_client
        self.model = model

    def extract_variables_and_edges(self, scenario_description: str) -> List[Tuple[str, str]]:
        """
        Use an LLM to extract a causal graph (edges) from a text scenario.

        Returns:
        - List of edges as (cause, effect) pairs
        """
        prompt = f"""
You're a causal inference modeler.

Read the following scenario and extract causal relationships in the form of edges (A -> B).

Respond only with a list of pairs like:
(A, B)
(B, C)

Scenario:
{scenario_description.strip()}
        """.strip()

        response = self.llm_client.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return self._parse_edges(response.choices[0].message["content"])

    def _parse_edges(self, raw_text: str) -> List[Tuple[str, str]]:
        """
        Parse raw LLM output to extract (A, B) pairs.
        """
        pattern = r"\(([^,]+),\s*([^)]+)\)"
        return re.findall(pattern, raw_text)
