"""Extension point: named attack categories for injection callers. Nothing in the run path imports this yet.

Where later tooldesk tasks plug in:
- B1 reads a persona's `attack_category` key, looks it up in ATTACK_CATEGORIES, and reports the attack
  success rate separately from the pass rate. The categories come from load_from_tooldesk().
- Persona files and run records that carry the new keys need schema version 2.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AttackCategory:
    name: str
    description: str
    examples: list[str] = field(default_factory=list)


ATTACK_CATEGORIES: dict[str, AttackCategory] = {}


def register(category: AttackCategory) -> AttackCategory:
    if not isinstance(category, AttackCategory):
        raise TypeError(f"expected an AttackCategory, got {type(category).__name__}")
    if category.name in ATTACK_CATEGORIES:
        raise ValueError(f"attack category {category.name!r} is already registered")
    ATTACK_CATEGORIES[category.name] = category
    return category


def load_from_tooldesk() -> list[AttackCategory]:
    raise NotImplementedError("tooldesk is not available")
