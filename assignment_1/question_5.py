import random

K = 2000
PATCH = 100
EPSILON = 0.1
ARM_REWARDS = [
    (-1.0, 4.0),
    ( 2.0, 6.0),
    (-2.0, 3.0),
    ( 5.0, 9.0),
    (-3.0, 5.0),
    ( 1.0, 6.0),
]
ideal_actions = [10.0 for _ in ARM_REWARDS]
action_count = [0 for _ in ARM_REWARDS]

ALPHA = 0.05
def new_average(old_average: float, reward: float) -> float:
    return old_average + ALPHA * (reward - old_average)

def update_actions(idx: int, reward: float) -> float:
    """Updates ``ideal_actions`` and ``action_count`` with the best
    reward/count"""
    ideal_actions[idx] = new_average(ideal_actions[idx], reward)
    action_count[idx] += 1
    return reward

def exploit() -> float:
    """Choose the best known action so far"""
    i, ideal = 0, ideal_actions[0]
    for j, v in enumerate(ideal_actions[1:], start=1):
        if v > ideal:
            ideal = v
            i = j
    reward = random.uniform(*ARM_REWARDS[i])
    return update_actions(i, reward)

def explore() -> float:
    """Choose a random action to perform"""
    i = random.choice(range(len(ARM_REWARDS)))
    reward = random.uniform(*ARM_REWARDS[i])
    return update_actions(i, reward)

overall_average = 0
for i in range(K//PATCH):
    actions = random.choices(
        (exploit, explore),
        weights=(1-EPSILON, EPSILON),
        k=PATCH,
    )
    for j, action in enumerate(actions, start=1):
        reward = action()
        overall_average = new_average(overall_average, reward)

    tokens = []
    for j, count in enumerate(action_count, start=1):
        per = 100 * count / ((i+1) * PATCH)
        tokens.append(f"arm({j})={per:.2f}%")
    print(" ".join(tokens))
    print(f"overall average = {overall_average:.2f}", end="\n\n")

    if i == 9:
        ARM_REWARDS[3] = (-4,3)
