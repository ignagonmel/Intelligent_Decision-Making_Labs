import numpy as np


transition_matrix = np.array([
    [0.10, 0.40, 0.40, 0.10],  # (0,0)
    [0.40, 0.10, 0.10, 0.40],  # (0,1)
    [0.40, 0.10, 0.10, 0.40],  # (1,0)
    [0.10, 0.40, 0.40, 0.10]   # (1,1)
])


def simulate_mc(transition_matrix, steps=10):
    state = 3  #(1,1)
    states = [state] 

    for _ in range(steps):
        state = np.random.choice([0, 1, 2, 3], p=transition_matrix[state])
        states.append(state)

    return states

#SIMULATION

states = simulate_mc(transition_matrix, steps=10)


state_names = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
visited = [state_names[s] for s in states]

print("States visited:", visited)
