"""Experiment configuration (YAML). See config.example.yaml for every field."""
import os
from dataclasses import dataclass, field

import yaml


@dataclass
class ModelSpec:
    id: str                    # short name used in outputs, e.g. qwen2.5-7b
    backend: str               # key into `backends`
    model: str                 # name the backend serves it under
    params_b: float            # total parameter count N, billions
    family: str = ""
    ladder: str = ""           # primary | secondary | comparator
    active_params_b: float = None   # N_a for MoE models; defaults to params_b
    version: str = ""          # exact version string / revision for the paper
    precision: str = "bf16"
    hardware: str = ""
    system_in_user: bool = False
    concurrency: int = None
    extra_body: dict = field(default_factory=dict)

    @property
    def n_active(self) -> float:
        return self.active_params_b or self.params_b


@dataclass
class Config:
    raw: dict
    path: str

    @property
    def items_csv(self):
        return self._rel(self.raw.get("items_csv", "llm_eval_prompt_matrix_300.csv"))

    @property
    def run_dir(self):
        return self._rel(os.path.join(self.raw.get("output_dir", "runs"), self.raw.get("run_name", "default")))

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 1234))

    @property
    def decoding(self) -> dict:
        d = {"max_tokens": 1024, "greedy_temperature": 0.0, "k_samples": 5,
             "sample_temperature": 0.7, "sample_top_p": 0.95}
        d.update(self.raw.get("decoding", {}))
        return d

    @property
    def backends(self) -> dict:
        return self.raw.get("backends", {})

    @property
    def models(self) -> list:
        return [ModelSpec(**m) for m in self.raw.get("models", [])]

    @property
    def judge(self) -> dict:
        j = {"model": "claude-opus-5-5", "effort": "medium", "concurrency": 8, "max_tokens": 16000}
        j.update(self.raw.get("judge", {}))
        return j

    @property
    def similarity(self) -> dict:
        s = {"embedding_model": "sentence-transformers/all-mpnet-base-v2", "bertscore_model": "roberta-large"}
        s.update(self.raw.get("similarity", {}))
        return s

    @property
    def analysis(self) -> dict:
        a = {"bootstrap_resamples": 10000, "alpha": 0.05}
        a.update(self.raw.get("analysis", {}))
        return a

    def _rel(self, p):
        return p if os.path.isabs(p) else os.path.join(os.path.dirname(os.path.abspath(self.path)), p)


def load_config(path: str) -> Config:
    with open(path) as f:
        return Config(yaml.safe_load(f), path)
