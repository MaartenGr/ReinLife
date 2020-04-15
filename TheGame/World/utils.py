from enum import IntEnum


class Actions(IntEnum):
    up = 0
    right = 1
    down = 2
    left = 3

    attack_up = 4
    attack_right = 5
    attack_down = 6
    attack_left = 7


class EntityTypes(IntEnum):
    empty = 0
    food = 1
    poison = 2
    agent = 3
    kin = 4
    super_berry = 5
