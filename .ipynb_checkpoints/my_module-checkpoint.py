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
import ray

@ray.remote
def some_function():
    return "Hello from the remote function!"

def local_function():
    return "This function is not remote."

