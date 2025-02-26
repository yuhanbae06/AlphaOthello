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
#     display_name: alphaothello
#     language: python
#     name: alphaothello
# ---

# +
from my_module2 import Module2
import ray

class Module():
    def __init__(self):
        self.module2 = Module2()

    @ray.remote
    def some_function(self):
        return self.module2.print_num()

    def some_function2(self):
        futures = [self.some_function.remote(self) for i in range(5)]
        memory_list = ray.get(futures)
        return memory_list
