import random

K = 20
ARM_REWARDS = (
    (-1, 4),
    ( 2, 6),
    (-2, 3),
    ( 5, 9),
    (-3, 5),
    ( 1, 6),
)

averages = 0
for _ in range(K):
    actions = random.choices(ARM_REWARDS, k=K)
    reward_sum = sum(
        map(
            lambda r: random.uniform(*r),
            actions,
        )
    )
    averages += reward_sum / K

print(averages/K)
