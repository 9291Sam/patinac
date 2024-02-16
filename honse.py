import secrets
from typing import List
from dataclasses import dataclass

@dataclass
class Weight:
    name: str
    weight: int

def random_weighted(weights: List[Weight]):
    return secrets.choice([w.name * w.weight for w in weights])

weights = [
    Weight("Honse1", 1),
    Weight("Honse2", 1)
]

for _ in range(100):
    choice = random_weighted(weights)
    print(choice, end=", ")
