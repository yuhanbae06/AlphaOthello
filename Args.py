# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: AlphaOthello
#     language: python
#     name: alphaothello
# ---

from dataclasses import dataclass, asdict


# +
@dataclass
class Args:
    C: float = 2
    num_searches: int = 60
    num_iterations: int = 3
    num_selfPlay_iterations: int = 500
    num_parallel_games: int = 100
    num_epochs: int = 4
    batch_size: int = 64
    temperature: float = 1.25
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.3

    @classmethod
    def exp0(cls):
        return cls()

    @classmethod
    def exp1(cls):
        return cls(num_searches=1, num_selfPlay_iterations=5, num_parallel_games=1)

    @classmethod
    def exp2(cls):
        return cls(num_searches=200, num_iterations=20, num_selfPlay_iterations=2500, num_parallel_games=125)
    
    def dict_(self):
        return asdict(self)

def get_args(exp_name="exp0"):
    return getattr(Args, exp_name)()
