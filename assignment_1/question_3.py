import random

K = 2000
PATCH = 100
EPSILON = 0.1
ARM_REWARDS = (
    (-1.0, 4.0),
    ( 2.0, 6.0),
    (-2.0, 3.0),
    ( 5.0, 9.0),
    (-3.0, 5.0),
    ( 1.0, 6.0),
)
ideal_actions = [lower for lower, _ in ARM_REWARDS]
action_count = [0 for _ in ARM_REWARDS]

def new_average(old_average: float, step: int, reward: float) -> float:
    return old_average + 1/(step+1) * (reward - old_average)

def update_actions(idx: int, reward: float, step: int) -> float:
    """Updates ``ideal_actions`` and ``action_count`` with the best
    reward/count"""
    ideal_actions[idx] = new_average(ideal_actions[idx], step, reward)
    action_count[idx] += 1
    return reward

def exploit(step: int) -> float:
    """Choose the best known action so far"""
    i, ideal = 0, ideal_actions[0]
    for j, v in enumerate(ideal_actions[1:], start=1):
        if v > ideal:
            ideal = v
            i = j
    reward = random.uniform(*ARM_REWARDS[i])
    return update_actions(i, reward, step)

def explore(step: int) -> float:
    """Choose a random action to perform"""
    i = random.choice(range(len(ARM_REWARDS)))
    reward = random.uniform(*ARM_REWARDS[i])
    return update_actions(i, reward, step)

overall_average = 0
for i in range(K//PATCH):
    actions = random.choices(
        (exploit, explore),
        weights=(1-EPSILON, EPSILON),
        k=PATCH,
    )
    for j, action in enumerate(actions, start=1):
        step = i * PATCH + j
        reward = action(step)
        overall_average = new_average(overall_average, step, reward)

    tokens = []
    for j, count in enumerate(action_count, start=1):
        per = 100 * count / ((i+1) * PATCH)
        tokens.append(f"arm({j})={per:.2f}%")
    print(" ".join(tokens))
    print(f"overall average = {overall_average:.2f}", end="\n\n")
