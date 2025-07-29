# causalllm/counterfactual_engine.py

from typing import Dict, Optional
import openai  # or use a pluggable LLM interface

class CounterfactualEngine:
    def __init__(self, llm_client=None, model="gpt-4"):
        """
        Initialize with an optional LLM client (like OpenAI, Anthropic, etc.)
        """
        self.llm_client = llm_client or openai
        self.model = model

    def simulate_counterfactual(
        self,
        context: str,
        factual: str,
        intervention: str,
        instruction: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Simulate a counterfactual based on input facts and intervention.

        Parameters:
        - context: Domain or background info (e.g., patient, system)
        - factual: Description of what actually happened
        - intervention: What you'd hypothetically change
        - instruction: Optional prompt tuning
        """
        prompt = self._build_prompt(context, factual, intervention, instruction)

        response = self.llm_client.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )

        return response.choices[0].message["content"].strip()

    def _build_prompt(
        self,
        context: str,
        factual: str,
        intervention: str,
        instruction: Optional[str]
    ) -> str:
        base_prompt = f"""
You are a causal reasoning expert.

Context:
{context.strip()}

Factual Scenario:
{factual.strip()}

Counterfactual Intervention:
{intervention.strip()}

Please describe the most plausible counterfactual outcome based on this change.
"""
        if instruction:
            base_prompt += f"\nAdditional Instruction: {instruction.strip()}"
        return base_prompt.strip()
