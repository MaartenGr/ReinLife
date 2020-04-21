from enum import IntEnum


class Actions(IntEnum):
    """ All possible actions """
    up = 0
    right = 1
    down = 2
    left = 3

    attack_up = 4
    attack_right = 5
    attack_down = 6
    attack_left = 7


class EntityTypes(IntEnum):
    """ The types of entities that are currently possible.
    NOTE: All entities except kin cannot occupy the same space"""

    empty = 0
    food = 1
    poison = 2
    agent = 3
    kin = 4
    super_food = 5
