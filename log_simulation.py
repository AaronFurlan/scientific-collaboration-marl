import json
import os
from pathlib import Path
from typing import Dict, List


class SimLog:
    def __init__(
        self, logs_dir: str, action_path: str, observation_path: str, project_path: str
    ) -> None:
        self.logs_dir = os.path.join(os.path.dirname(__file__), logs_dir)
        os.makedirs(logs_dir, exist_ok=True)
        self.action_path = os.path.join(logs_dir, action_path)
        self.observation_path = os.path.join(logs_dir, observation_path)
        self.project_path = os.path.join(logs_dir, project_path)

    def start(self):
        # Delete existing files if they exist
        for file_path in [self.action_path, self.observation_path, self.project_path]:
            if os.path.exists(file_path):
                os.remove(file_path)

    def log_action(self, action: Dict):
        # Append JSONL row
        with open(self.action_path, "a") as jf:
            jf.write(json.dumps(convert_numpy(action)) + "\n")

    def log_observation(self, obs: Dict):
        # Append JSONL row
        with open(self.observation_path, "a") as jf:
            jf.write(json.dumps(convert_numpy(obs)) + "\n")

    def log_projects(self, projects: List):
        with open(self.project_path, "w") as jf:
            json.dump([convert_numpy(p.to_dict()) for p in projects], jf, indent=2)


# Convert numpy arrays to lists for JSON serialization (handles nested structures)
def convert_numpy(obj):
    if hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj
