"""
This module contains code to pre-process the randomly generated data.
The overall goal is to improve the final model's performance by improving the data it is trained on.

Improving the data:
- reduce the complete duplicates
    - it happens that the exact same datapoint appears multiple times
- maintain the relative order, but scale down the most popular entity types
- maintain the relative order, but scale down the most popular entities
- maintain the relative order, but scale down the most popular rules
"""