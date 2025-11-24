import numpy as np
from hmmlearn import hmm

# EXERCISE 4: Define HMM Model for robot


# Hidden States
# 0 = alone -> sleeping
# 1 = idle -> calling user
# 2 = interacting -> playing a game
model = hmm.CategoricalHMM(n_components=3)

# Initial Probabilities
# I assume robot is more likely to start alone
model.startprob_ = np.array([0.7, 0.2, 0.1])

# Transition Matrix
model.transmat_ = np.array([
    [0.75, 0.20, 0.05],  # alone → [alone, idle, interacting]
    [0.30, 0.50, 0.20],  # idle → [alone, idle, interacting]
    [0.10, 0.30, 0.60]   # interacting → [alone, idle, interacting]
])

# OBSERBATIONS
# 3 sensors, binary → 2×2×2 = 8 obs
#
# Encoding:
# 0 = (no body, no face, no voice)
# 1 = (no body, no face, voice)
# 2 = (no body, face, no voice)
# 3 = (no body, face, voice)
# 4 = (body, no face, no voice)
# 5 = (body, no face, voice)
# 6 = (body, face, no voice)
# 7 = (body, face, voice)
# ============================================================

# Emition Matrix
model.emissionprob_ = np.array([
    # 0    1     2     3     4     5     6     7
    [0.70, 0.10, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01],  # alone
    [0.05, 0.05, 0.10, 0.10, 0.20, 0.20, 0.20, 0.10],  # idle
    [0.01, 0.01, 0.05, 0.10, 0.10, 0.15, 0.20, 0.38]   # interacting
])


# Printing states names
def print_states(seq):
    mapping = {0: "alone", 1: "idle", 2: "interacting"}
    print("Hidden States:", [mapping[s] for s in seq])



# EXERCISE 5: Simulation

observations = np.array([[4, 6, 7, 7, 6]]).T

print("Simulated Observations:", observations.ravel())

# Predict states sequency for obs
hidden_states = model.predict(observations)
print_states(hidden_states)

# Most probable future state
last_state = hidden_states[-1]
future_state = np.argmax(model.transmat_[last_state])
print("Most probable future state:", future_state)

# Most probable future observation
future_observation = np.argmax(model.emissionprob_[future_state])
print("Most probable future observation:", future_observation)


# EXERCISE 6: Simulate 10 iterations

mapping = {0: "alone", 1: "idle", 2: "interacting"}

actions = {
    0: "Maggie is sleeping",
    1: "Maggie calling User",
    2: "Maggie playing with User"
}

for i in range(10):
    print(f"\nIteration {i+1}")

    #I assumed Maggie will always start alone for simplicity and more likely scenario
    if i == 0:
        real_state = 0
    else:
        real_state = np.random.choice([0,1,2], p=model.transmat_[real_state])

    obs = np.random.choice(8, p=model.emissionprob_[real_state])
    print("Observation:", obs)

    
    predicted_state = model.predict(np.array([[obs]]))[0]
    print("State:", mapping[predicted_state])

    print(actions[predicted_state])

print("\n=====SIMULATION COMPLETED=====")
