# causalllm/do_operator.py

from typing import Dict, Optional


class DoOperatorSimulator:
    """
    Simulates the effect of 'do(X=x)' interventions by modifying context and variables.
    """

    def __init__(self, base_context: str, variables: Dict[str, str]):
        """
        Parameters:
        - base_context: A paragraph describing the initial (observational) setup
        - variables: A dictionary of variable names and their current values
        """
        self.base_context = base_context
        self.variables = variables.copy()

    def intervene(self, interventions: Dict[str, str]) -> str:
        """
        Apply a do() intervention by replacing variable values in the context.

        Parameters:
        - interventions: Dict like {"Treatment": "New_UI"}

        Returns:
        - Modified context reflecting the intervention
        """
        modified_context = self.base_context

        for var, new_val in interventions.items():
            if var not in self.variables:
                raise ValueError(f"Variable '{var}' not in base context.")
            original_val = self.variables[var]
            modified_context = modified_context.replace(original_val, new_val)
            self.variables[var] = new_val  # update for future chaining

        return modified_context

    def generate_do_prompt(
        self,
        interventions: Dict[str, str],
        question: Optional[str] = None
    ) -> str:
        """
        Construct a causal prompt reflecting the do() intervention.

        Parameters:
        - interventions: Dict of variable interventions
        - question: Optional follow-up question to be answered by an LLM

        Returns:
        - Full prompt string
        """
        modified_context = self.intervene(interventions)
        intervention_desc = ", ".join([f"{k} := {v}" for k, v in interventions.items()])
        prompt = f"""
You are a causal inference model.

Base scenario:
{self.base_context}

Intervention applied:
do({intervention_desc})

Resulting scenario:
{modified_context}

{question if question else "What is the expected impact of this intervention?"}
"""
        return prompt.strip()
