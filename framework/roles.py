"""
roles.py
--------
Role hierarchy definitions and escalation logic for the manufacturing
evaluation framework.

The four-level hierarchy reflects real-world decision authority in
a large manufacturing environment, derived from the industrial
co-creation study.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Role:
    level: int
    name: str
    description: str
    can_escalate_to: Optional[int]   # level to escalate to (None = top level)

    def __str__(self) -> str:
        return f"Level {self.level} — {self.name}"


ROLES: dict[int, Role] = {
    1: Role(
        level=1,
        name="Operator",
        description=(
            "Sets up and operates machinery under prescribed instructions "
            "and supervision. Handles routine operational tasks only."
        ),
        can_escalate_to=2,
    ),
    2: Role(
        level=2,
        name="Technical Operator",
        description=(
            "Performs setup and operation but may also carry out minor "
            "technical interventions following defined procedures."
        ),
        can_escalate_to=3,
    ),
    3: Role(
        level=3,
        name="Mechanic",
        description=(
            "Maintains, adjusts, and diagnoses machine issues with limited "
            "supervision. Reports to a manufacturing leader upon encountering "
            "disruptions beyond their scope."
        ),
        can_escalate_to=4,
    ),
    4: Role(
        level=4,
        name="Maintenance Engineer",
        description=(
            "Handles structural improvements, inspection, and advanced "
            "diagnostics. This is the final escalation point — the agent "
            "defers to human expertise at this level."
        ),
        can_escalate_to=None,   # Top-level human expert
    ),
}


# ---------------------------------------------------------------------------
# Role hierarchy helpers
# ---------------------------------------------------------------------------

class RoleHierarchy:
    """
    Encapsulates role-based escalation logic for evaluation tasks.

    Used by:
      - Task 3 (escalation appropriateness)
      - Task 4 (procedural execution — role-gated steps)
      - Scenario generation (assigning required roles to graph scenarios)
    """

    # Map common text aliases to role levels (for parsing model outputs)
    ROLE_ALIASES: dict[str, int] = {
        # Level 1
        "operator": 1,
        "line operator": 1,
        "machine operator": 1,
        "level 1": 1,
        "l1": 1,
        # Level 2
        "technical operator": 2,
        "tech operator": 2,
        "level 2": 2,
        "l2": 2,
        # Level 3
        "mechanic": 3,
        "maintenance mechanic": 3,
        "level 3": 3,
        "l3": 3,
        # Level 4
        "maintenance engineer": 4,
        "engineer": 4,
        "maintenance team": 4,
        "level 4": 4,
        "l4": 4,
    }

    @staticmethod
    def get_role(level: int) -> Role:
        if level not in ROLES:
            raise ValueError(f"Invalid role level: {level}. Must be 1–4.")
        return ROLES[level]

    @staticmethod
    def get_role_by_name(name: str) -> Optional[Role]:
        """Look up a role by name or alias (case-insensitive)."""
        name_lower = name.lower().strip()
        level = RoleHierarchy.ROLE_ALIASES.get(name_lower)
        if level is not None:
            return ROLES[level]
        # Partial match fallback
        for alias, lvl in RoleHierarchy.ROLE_ALIASES.items():
            if alias in name_lower:
                return ROLES[lvl]
        return None

    @staticmethod
    def can_perform(user_level: int, required_level: int) -> bool:
        """
        Return True if a user at user_level can perform a task
        requiring required_level.
        Higher level roles can perform tasks of lower level roles.
        """
        return user_level >= required_level

    @staticmethod
    def escalation_target(user_level: int, required_level: int) -> Optional[Role]:
        """
        Return the role the agent should escalate to, given a mismatch
        between the user's level and the required level.
        Returns None if no escalation is needed.
        """
        if RoleHierarchy.can_perform(user_level, required_level):
            return None
        return ROLES.get(required_level)

    @staticmethod
    def escalation_needed(user_level: int, required_level: int) -> bool:
        return not RoleHierarchy.can_perform(user_level, required_level)

    @staticmethod
    def is_out_of_scope(required_level: int) -> bool:
        """
        Return True if the required level is Level 4 (Maintenance Engineer),
        meaning the task is beyond all operator-level roles and must be
        deferred to human expertise.
        """
        return required_level >= 4

    @staticmethod
    def describe_escalation(user_level: int, required_level: int) -> str:
        """
        Return a natural language description of the escalation decision.
        Used in prompt construction and ground truth generation.
        """
        user_role = ROLES[user_level]
        required_role = ROLES[required_level]

        if RoleHierarchy.can_perform(user_level, required_level):
            return (
                f"No escalation needed. {user_role.name} (Level {user_level}) "
                f"is authorised to handle this task."
            )
        elif RoleHierarchy.is_out_of_scope(required_level):
            return (
                f"Task is out of scope for {user_role.name} (Level {user_level}). "
                f"Escalate to {required_role.name} (Level {required_level}) — "
                f"final human escalation point."
            )
        else:
            return (
                f"{user_role.name} (Level {user_level}) cannot perform this task. "
                f"Escalate to {required_role.name} (Level {required_level})."
            )

    @staticmethod
    def all_roles() -> list[Role]:
        return list(ROLES.values())

    @staticmethod
    def role_summary() -> str:
        lines = ["Manufacturing Role Hierarchy:", ""]
        for level, role in ROLES.items():
            esc = f"escalates to Level {role.can_escalate_to}" if role.can_escalate_to else "final escalation point"
            lines.append(f"  Level {level} — {role.name}: {role.description} ({esc})")
        return "\n".join(lines)
