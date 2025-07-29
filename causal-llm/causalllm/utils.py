"""
Utility functions for CausalLLM
"""

import json
import yaml


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_yaml(file_path):
    """Load YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data, file_path):
    """Save data to a YAML file."""
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
