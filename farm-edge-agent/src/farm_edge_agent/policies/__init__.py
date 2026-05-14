"""Policy backends — programs that map (obs, prompt) → actions.

Two backends are wired:

- ``GptSkillPolicy``: the GPT-decomposer + hand-coded skill library used
  in the rest of this demo. Cheap, deterministic, runs on CPU.
- ``Pi05Policy``: Physical Intelligence's π0.5 VLA model, served on a
  remote GPU via Modal. Takes camera + state + prompt, returns joint
  deltas at the model's control rate.
"""

from farm_edge_agent.policies.pi05 import Pi05Policy, Pi05Result

__all__ = ["Pi05Policy", "Pi05Result"]
