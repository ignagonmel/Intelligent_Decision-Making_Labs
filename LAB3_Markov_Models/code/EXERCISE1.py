import numpy as np
# Define transition matrix
transition_matrix = np.array([[0.7, 0.2, 0.1], # Probabilities for state A: [A->A, A->B, A->C]
    [0.3, 0.4, 0.3], # Probabilities for state B: [B->A, B->B, B->C]
    [0.1, 0.3, 0.6]]) # Probabilities for state C: [C->A, C->B, C->C]
# Simulate Markov Chain for a number of steps
def simulate_mc(transition_matrix, steps=10):
    state = 0 # Start in state A
    states = [state] # Define a list to save following states
# Loop for some steps
    for _ in range(steps):
        state = np.random.choice([0, 1, 2], p=transition_matrix[state])
        states.append(state)
    return states
# Simulate the chain
states = simulate_mc(transition_matrix, steps=15)

state_names = ['A', 'B', 'C']
visited = [state_names[s] for s in states]

print("States visited:", visited)