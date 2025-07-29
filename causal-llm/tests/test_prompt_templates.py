# tests/test_prompt_templates.py

import unittest
from causalllm.prompt_templates import PromptTemplates

class TestPromptTemplates(unittest.TestCase):

    def test_treatment_effect_prompt(self):
        prompt = PromptTemplates.treatment_effect_estimation(
            context="Experiment on users.",
            treatment="New design",
            outcome="Engagement"
        )
        self.assertIn("Treatment Variable", prompt)
        self.assertIn("Outcome Variable", prompt)

    def test_chain_of_thought_prompt(self):
        steps = {1: "Consider weather.", 2: "Consider umbrella."}
        prompt = PromptTemplates.causal_chain_of_thought("Avoid getting wet", steps)
        self.assertIn("Step 1", prompt)
        self.assertIn("Avoid getting wet", prompt)

if __name__ == "__main__":
    unittest.main()
