# tests/test_counterfactual_engine.py

import unittest
from causalllm.counterfactual_engine import CounterfactualEngine

class DummyLLMClient:
    def ChatCompletion_create(self, model, messages, temperature):
        return type("Response", (), {
            "choices": [type("Choice", (), {"message": {"content": "Simulated counterfactual outcome."}})]
        })()

    # Alias method for compatibility
    ChatCompletion = type("ChatCompletion", (), {"create": ChatCompletion_create})

class TestCounterfactualEngine(unittest.TestCase):

    def test_simulate_counterfactual(self):
        engine = CounterfactualEngine(llm_client=DummyLLMClient())
        result = engine.simulate_counterfactual(
            context="Test context.",
            factual="Factual scenario.",
            intervention="Changed variable."
        )
        self.assertIn("Simulated counterfactual outcome", result)

if __name__ == "__main__":
    unittest.main()
