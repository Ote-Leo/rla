#set page("a4")
#set par(justify: true)

#align(center, text(17pt)[ *Robot Learning -- Exercise Sheet 1* ])
#grid(
  columns: (1fr, 1fr),
  align(center)[
    Omar Elsebaey \
    50357345 \
    #link("mailto:s28oelse@uni-bonn.de")
  ],
  align(center)[
    Martin Halbfas \
    3329972 \
    #link("mailto:s14mhalb@uni-bonn.de")
  ],
)

= Question 1 <q1>

#rect(width: 100%, [
  Given is a six-armed bandit, as introduced in the lecture.

  - The first arm shall sample its reward uniformly from the interval $[-1,4)$.
  - The second arm shall sample its reward uniformly from $[2,6)$.
  - The third arm shall sample its reward uniformly from the interval $[-2,3)$.
  - The fourth arm shall sample its reward uniformly from $[5,9)$.
  - The fifth arm shall sample its reward uniformly from $[-3,5)$.
  - The sixth arm shall sample its reward uniformly from $[1,6)$.

  What is the expected reward when actions are chosen uniformly?
])

For a *uniform distribution* $cal(U)(a,b)$, the *expected value* (mean) is
given by:

$
  EE(X) = (a + b) / 2
$

This is because the uniform distribution is symmetric --- it's "center"
(midpoint) is its average. So when we want to compute the expected reward of
pulling, say Arm#sub("1") which gives a reward from $cal(U)(-1,4)$, the average
reward over time (expected reward) is:

$
  EE("Reward") = (-1 + 4) / 2 = 1.5
$

Applying this to each arm:

- Arm#sub("1"): $[-1, 4) => (-1+4)/2 = 1.5$
- Arm#sub("2"): $[ 2, 6) => (2+6)/2  = 4$
- Arm#sub("3"): $[-2, 3) => (-2+3)/2 = 0.5$
- Arm#sub("4"): $[ 5, 9) => (5+9)/2  = 7$
- Arm#sub("5"): $[-3, 5) => (-3+5)/2 = 1$
- Arm#sub("6"): $[ 1, 6) => (1+6)/2  = 3.5$

Since actions are chosen *uniformly*, the overall expected reward is the
average of the expected values:

$
  1/6(1.5+4+0.5+7+1+3.5) = 17.5/6 approx 2.9167
$

#pagebreak()

= Question 2 <q2>

#rect(width: 100%, [
  Implement the six-armed bandit from #link(<q1>)[1.1)] and compute the sample
  average reward for 20 uniformly chosen actions!

  Compare this to your expectation from #link(<q1>)[1.1)]!
])

Using the following (Python3) implementation

```python
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
    reward_sum = sum(map(lambda r: random.uniform(*r), actions))
    averages += reward_sum / K

print(averages/K)
```

The output of this program is `2.828300482608005`, which quickly approximates
to the expected output we had in #link(<q1>)[Question 1]

#pagebreak()

= Question 3 <q3>

#rect(width: 100%, [
  Initialize $Q(a_i)=0$ and chose $2000$ actions according to an
  $epsilon$-greedy selection strategy ($epsilon=0.1$)!

  Update your action values by computing the sample average reward of each
  action recursively!

  For every $100$ actions show the percentage of choosing arm#sub("1"),
  arm#sub("2"), arm#sub("3"), arm#sub("4"), arm#sub("5"), and arm#sub("6") as
  well as the resulting average reward!
])

Starting by setting up the global variables

```python
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
```

where `ideal_actions` will be used to track the each arm overall average and
choose the best accordingly in the exploitation step and using the
`action_count` to keep track of the selected action during runtime.

Setting the average updating function

```python
def new_average(old_average: float, step: int, reward: float) -> float:
    return old_average + 1/(step+1) * (reward - old_average)
```

following the rules of

$
  "NewEstimate" = "OldEstimate" + "StepSize"["Target" - "OldEstimate"]
$

Having the decision functions of

```python
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
```

And running the program in main loop

```python
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
```

where we're always choosing to either `exploit` or `explore` based on
the `EPSILON` destribution ($90%$ exploiting and exploring $10%$ of
the times)

After running the program we end up with the following output

```
arm(1)=2.00% arm(2)=3.00% arm(3)=1.00% arm(4)=91.00% arm(5)=0.00% arm(6)=3.00%
overall average = 6.53

arm(1)=1.00% arm(2)=3.00% arm(3)=1.50% arm(4)=91.50% arm(5)=1.00% arm(6)=2.00%
overall average = 6.58

arm(1)=1.00% arm(2)=2.33% arm(3)=1.00% arm(4)=93.00% arm(5)=1.33% arm(6)=1.33%
overall average = 6.70

arm(1)=1.25% arm(2)=2.00% arm(3)=1.50% arm(4)=92.00% arm(5)=1.75% arm(6)=1.50%
overall average = 6.59

arm(1)=1.40% arm(2)=1.60% arm(3)=1.20% arm(4)=92.00% arm(5)=2.20% arm(6)=1.60%
overall average = 6.60

arm(1)=1.17% arm(2)=1.50% arm(3)=1.33% arm(4)=92.17% arm(5)=2.33% arm(6)=1.50%
overall average = 6.57

arm(1)=1.29% arm(2)=2.00% arm(3)=1.43% arm(4)=91.86% arm(5)=2.14% arm(6)=1.29%
overall average = 6.57

arm(1)=1.38% arm(2)=2.00% arm(3)=1.38% arm(4)=91.50% arm(5)=2.25% arm(6)=1.50%
overall average = 6.57

arm(1)=1.22% arm(2)=2.22% arm(3)=1.56% arm(4)=91.56% arm(5)=2.11% arm(6)=1.33%
overall average = 6.56

arm(1)=1.50% arm(2)=2.10% arm(3)=1.40% arm(4)=91.50% arm(5)=2.10% arm(6)=1.40%
overall average = 6.57

arm(1)=1.82% arm(2)=2.00% arm(3)=1.36% arm(4)=91.55% arm(5)=1.91% arm(6)=1.36%
overall average = 6.56

arm(1)=2.00% arm(2)=2.00% arm(3)=1.50% arm(4)=91.50% arm(5)=1.75% arm(6)=1.25%
overall average = 6.55

arm(1)=2.08% arm(2)=1.92% arm(3)=1.38% arm(4)=91.69% arm(5)=1.69% arm(6)=1.23%
overall average = 6.55

arm(1)=2.14% arm(2)=1.86% arm(3)=1.36% arm(4)=91.64% arm(5)=1.64% arm(6)=1.36%
overall average = 6.56

arm(1)=2.00% arm(2)=1.80% arm(3)=1.27% arm(4)=92.00% arm(5)=1.60% arm(6)=1.33%
overall average = 6.58

arm(1)=2.06% arm(2)=1.75% arm(3)=1.62% arm(4)=91.56% arm(5)=1.62% arm(6)=1.38%
overall average = 6.56

arm(1)=2.00% arm(2)=1.65% arm(3)=1.65% arm(4)=91.76% arm(5)=1.65% arm(6)=1.29%
overall average = 6.58

arm(1)=2.11% arm(2)=1.72% arm(3)=1.67% arm(4)=91.44% arm(5)=1.61% arm(6)=1.44%
overall average = 6.58

arm(1)=2.00% arm(2)=1.74% arm(3)=1.63% arm(4)=91.53% arm(5)=1.58% arm(6)=1.53%
overall average = 6.57

arm(1)=1.95% arm(2)=1.75% arm(3)=1.55% arm(4)=91.65% arm(5)=1.55% arm(6)=1.55%
overall average = 6.58
```

Which also agrees with the results show in #link(<q1>)[Question 1], since
Arm#sub("4") has the highest overall average and since we're spending $90%$ of
our trials exploiting the algorithm quickly learns to always choose from
Arm#sub("4"), raising the overall average to a whopping $6.58$ rather than the
$2.82$ we had in #link(<q2>)[Question 2].

It's alive 🙂


#pagebreak()

= Question 4 <q4>

#rect(width: 100%, [
  Redo the experiment, but after $1000$ steps sample the rewards of the fourth
  arm uniformly from $[-4,3)$!

  Compare updating action values by computing the sample average reward of each
  action recursively (as done in #link(<q3>)[1.3]) with using a constant
  learning rate $alpha=0.05$!

  For every $100$ actions show the percentage of choosing arm#sub("1"),
  arm#sub("2"), arm#sub("3"), arm#sub("4"), arm#sub("5"), and arm#sub("6") as
  well as the resulting average reward!
])

By updating each of

/ ```python ARM_REWARDS```:

```diff
- ARM_REWARDS = (
+ ARM_REWARDS = [
  ...
- )
+ ]
```

/ ```python new_average(old_average: float, step: int, reward: float) -> float```:

```diff
- def new_average(old_average: float, step: int, reward: float) -> float:
-    return old_average + 1/(step+1) * (reward - old_average)
+ ALPHA = 0.05
+ def new_average(old_average: float, reward: float) -> float:
+    return old_average + ALPHA * (reward - old_average)
```

/ ```python update_actions(idx: int, reward: float, step: int) -> float```:

```diff
- def update_actions(idx: int, reward: float, step: int) -> float:
+ def update_actions(idx: int, reward: float) -> float:
  ...
-    ideal_actions[idx] = new_average(ideal_actions[idx], reward, step)
+    ideal_actions[idx] = new_average(ideal_actions[idx], reward)
  ...

```

/ ```python exploit(step: int) -> float```:

```diff
- def exploit(step: int) -> float:
+ def exploit() -> float:
  ...
-    return update_actions(i, reward, step)
+    return update_actions(i, reward)
```
/ ```python explore(step: int) -> float```:

```diff
- def explore(step: int) -> float:
+ def explore() -> float:
  ...
-    return update_actions(i, reward, step)
+    return update_actions(i, reward)
```
And updating the body of the main loop with

```diff
overall_average = 0
for i in range(K//PATCH):
  ...
    for j, action in enumerate(actions, start=1):
-        step = i * PATCH + j
-        reward = action(step)
-        overall_average = new_average(overall_average, step, reward)
+        reward = action()
+        overall_average = new_average(overall_average, reward)

    tokens = []
    for j, count in enumerate(action_count, start=1):
        per = 100 * count / ((i+1) * PATCH)
        tokens.append(f"arm({j})={per:.2f}%")
    print(" ".join(tokens))
    print(f"overall average = {overall_average:.2f}", end="\n\n")
+    if i == 9:
+        ARM_REWARDS[3] = (-4,3)
```

We get the following output

```
arm(1)=2.00% arm(2)=0.00% arm(3)=1.00% arm(4)=93.00% arm(5)=3.00% arm(6)=1.00%
overall average = 6.48

arm(1)=1.50% arm(2)=0.50% arm(3)=0.50% arm(4)=94.00% arm(5)=2.50% arm(6)=1.00%
overall average = 6.48

arm(1)=1.67% arm(2)=0.33% arm(3)=0.33% arm(4)=94.00% arm(5)=2.67% arm(6)=1.00%
overall average = 6.83

arm(1)=1.50% arm(2)=0.75% arm(3)=0.75% arm(4)=93.25% arm(5)=3.00% arm(6)=0.75%
overall average = 6.97

arm(1)=1.20% arm(2)=1.00% arm(3)=0.80% arm(4)=93.40% arm(5)=2.60% arm(6)=1.00%
overall average = 6.74

arm(1)=1.00% arm(2)=1.50% arm(3)=0.67% arm(4)=93.33% arm(5)=2.50% arm(6)=1.00%
overall average = 6.75

arm(1)=1.14% arm(2)=1.43% arm(3)=0.71% arm(4)=93.57% arm(5)=2.14% arm(6)=1.00%
overall average = 6.70

arm(1)=1.50% arm(2)=1.62% arm(3)=0.75% arm(4)=93.12% arm(5)=2.00% arm(6)=1.00%
overall average = 6.42

arm(1)=1.67% arm(2)=1.44% arm(3)=0.67% arm(4)=93.22% arm(5)=2.11% arm(6)=0.89%
overall average = 6.41

arm(1)=1.50% arm(2)=1.50% arm(3)=0.70% arm(4)=93.50% arm(5)=2.00% arm(6)=0.80%
overall average = 7.07

arm(1)=1.55% arm(2)=7.91% arm(3)=1.00% arm(4)=86.64% arm(5)=1.91% arm(6)=1.00%
overall average = 3.88

arm(1)=1.58% arm(2)=14.92% arm(3)=1.00% arm(4)=79.67% arm(5)=1.83% arm(6)=1.00%
overall average = 3.37

arm(1)=1.46% arm(2)=21.08% arm(3)=1.00% arm(4)=73.69% arm(5)=1.69% arm(6)=1.08%
overall average = 3.90

arm(1)=1.57% arm(2)=26.14% arm(3)=1.00% arm(4)=68.50% arm(5)=1.71% arm(6)=1.07%
overall average = 3.71

arm(1)=1.53% arm(2)=30.60% arm(3)=1.00% arm(4)=63.93% arm(5)=1.87% arm(6)=1.07%
overall average = 3.66

arm(1)=1.62% arm(2)=34.31% arm(3)=0.94% arm(4)=60.19% arm(5)=1.94% arm(6)=1.00%
overall average = 3.17

arm(1)=1.82% arm(2)=37.65% arm(3)=0.94% arm(4)=56.82% arm(5)=1.82% arm(6)=0.94%
overall average = 3.55

arm(1)=1.83% arm(2)=40.67% arm(3)=1.00% arm(4)=53.72% arm(5)=1.78% arm(6)=1.00%
overall average = 3.85

arm(1)=1.84% arm(2)=43.21% arm(3)=1.11% arm(4)=51.11% arm(5)=1.68% arm(6)=1.05%
overall average = 3.55

arm(1)=1.90% arm(2)=45.45% arm(3)=1.30% arm(4)=48.60% arm(5)=1.70% arm(6)=1.05%
overall average = 3.72
```

Which shows the algorithms adaptability since during midway Arm#sub("4") isn't
no longer the action with the best overall average ($EE("Arm"_4) = (-4 + 3) / 2
= -0.5$) the second best canidate according to #link(<q1>)[Question 1] is
Arm#sub("2") with an overall average of $4$, which explains the increase in
Arm#sub("2") usage.

It's alive 🙂

#pagebreak()

= Question 5 <q5>

#rect(width: 100%, [
  Modify the experiment from #link(<q4>)[1.4)] by using an optimistic
  initialization $Q(a_i)=10$ and a greedy action selection strategy, still
  using a constant learning rate $alpha=0.05$!

  For every $100$ actions show the percentage of choosing arm#sub("1"),
  arm#sub("2"), arm#sub("3"), arm#sub("4"), arm#sub("5"), and arm#sub("6") as
  well as the resulting average reward!

  Compare this to your result from #link(<q4>)[1.4)]!
])

By updating `ideal_actions` to

```diff
- ideal_actions = [lower for lower, _ in ARM_REWARDS]
+ ideal_actions = [10.0 for _ in ARM_REWARDS]
```

We get the following

```
arm(1)=11.00% arm(2)=14.00% arm(3)=7.00% arm(4)=48.00% arm(5)=8.00% arm(6)=12.00%
overall average = 5.44

arm(1)=7.00% arm(2)=9.00% arm(3)=4.00% arm(4)=67.50% arm(5)=4.50% arm(6)=8.00%
overall average = 6.16

arm(1)=5.33% arm(2)=6.33% arm(3)=4.00% arm(4)=73.33% arm(5)=4.67% arm(6)=6.33%
overall average = 6.64

arm(1)=4.75% arm(2)=5.25% arm(3)=3.25% arm(4)=76.75% arm(5)=4.75% arm(6)=5.25%
overall average = 6.28

arm(1)=4.20% arm(2)=4.60% arm(3)=3.20% arm(4)=79.00% arm(5)=4.20% arm(6)=4.80%
overall average = 6.29

arm(1)=3.50% arm(2)=4.50% arm(3)=3.17% arm(4)=80.50% arm(5)=4.33% arm(6)=4.00%
overall average = 6.58

arm(1)=3.29% arm(2)=4.29% arm(3)=3.00% arm(4)=81.86% arm(5)=3.86% arm(6)=3.71%
overall average = 6.39

arm(1)=3.12% arm(2)=4.00% arm(3)=3.12% arm(4)=82.75% arm(5)=3.75% arm(6)=3.25%
overall average = 6.00

arm(1)=2.78% arm(2)=3.78% arm(3)=3.11% arm(4)=83.78% arm(5)=3.44% arm(6)=3.11%
overall average = 6.60

arm(1)=2.80% arm(2)=3.40% arm(3)=2.90% arm(4)=84.80% arm(5)=3.20% arm(6)=2.90%
overall average = 6.54

arm(1)=2.55% arm(2)=7.36% arm(3)=2.82% arm(4)=78.45% arm(5)=3.00% arm(6)=5.82%
overall average = 3.81

arm(1)=2.33% arm(2)=13.50% arm(3)=2.75% arm(4)=72.08% arm(5)=2.83% arm(6)=6.50%
overall average = 4.09

arm(1)=2.15% arm(2)=19.62% arm(3)=2.77% arm(4)=66.54% arm(5)=2.85% arm(6)=6.08%
overall average = 3.88

arm(1)=2.07% arm(2)=24.86% arm(3)=2.71% arm(4)=62.00% arm(5)=2.64% arm(6)=5.71%
overall average = 3.52

arm(1)=2.20% arm(2)=29.33% arm(3)=2.60% arm(4)=57.93% arm(5)=2.53% arm(6)=5.40%
overall average = 3.55

arm(1)=2.12% arm(2)=33.19% arm(3)=2.56% arm(4)=54.44% arm(5)=2.62% arm(6)=5.06%
overall average = 3.42

arm(1)=2.00% arm(2)=36.82% arm(3)=2.41% arm(4)=51.29% arm(5)=2.59% arm(6)=4.88%
overall average = 4.07

arm(1)=1.89% arm(2)=39.72% arm(3)=2.39% arm(4)=48.44% arm(5)=2.67% arm(6)=4.89%
overall average = 3.96

arm(1)=1.79% arm(2)=42.58% arm(3)=2.42% arm(4)=45.89% arm(5)=2.53% arm(6)=4.79%
overall average = 3.72

arm(1)=1.90% arm(2)=44.95% arm(3)=2.35% arm(4)=43.70% arm(5)=2.55% arm(6)=4.55%
overall average = 3.97
```

Which does exhibit a similar behaviour to #link(<q4>)[Question 4] answers,
(i.e. it starts exploit Arm#sub("2") midways) with an increase in the overall
average and an increase in the overall usage of different arms (aka
exploration).
