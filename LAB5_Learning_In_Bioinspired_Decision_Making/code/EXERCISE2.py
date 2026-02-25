import numpy as np
import random # Needed for np.random.choice

# Initialize the environment
GRID_SIZE = 5

# THE WORLD MAP WORKS AS FOLLOWS
# states S = {s_0, s_1, ..., s_24}
#
#     x    0    1    2    3    4
# y     0 s_0  s_1  s_2  s_3  s_4
#       1 s_5  s_6  s_7  s_8  s_9
#       2 s_10 s_11 s_12 s_13 s_14
#       3 s_15 s_16 s_17 s_18 s_19
#       4 s_20 s_21 s_22 s_23 s_24

# OBSTACLES = [(1, 2), (1, 1), (2, 1), (3, 1), (3, 3), (4, 3)]
OBSTACLES = [(2, 2)]

START_STATE = (0, 0)
# EXTENSION 1: Add new action 'interact' (index 5)
ACTION_MAP = {0: '↑', 1: '↓', 2: '←', 3: '→', 4: '🔌', 5: '🤝'}  # Actions = {up, down, left, right, charge, interact}
NUM_ACTIONS = len(ACTION_MAP) # CRITICAL FIX: Now 6 actions

MAX_ITER = 10000

# Q-learning parameters
LEARNING_RATE = 0.3
DISCOUNT_FACTOR = 0.7

# Epsilon parameter in the epsilon-greedy algorithm for action selection
EPSILON = 1

# Defining motivations and drives
# EXTENSION 2: Add socialization motivation and drive
MOTIVATIONS_MAP = {'recharge': 0, 'socialize': 1}
DRIVES_MAP = {'energy': 0, 'socialization': 1}
DRIVE_ENERGY_INCREMENT_RATE = 2
DRIVE_SOCIALIZATION_INCREMENT_RATE = 1 # Define a rate for the new drive
motivation_intensities = {'recharge': 0.0, 'socialize': 0.0}  # Initialization of motivations
drive_values = {'energy': 0.0, 'socialization': 0.0}  # Initialization of drives: from 0.0 to 100.0

# Locations of external stimuli
CHARGER_LOCATIONS = [(0, 3)]
# EXTENSION 3: Define People Locations
PEOPLE_LOCATIONS = [(3, 4), (4, 0)]

# Initialization of q-values in q-table
# CRITICAL FIX: Q-table dimension must now be (5, 5, 6)
q_table = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS)) 

# Returns the updated value for a motivation and its associated drive. This is executed in each iteration.
def update_motivation_drive(drive, increment, ext_stimuli=0):
    # Motivation = Drive + Stimuli 
    motivation = drive + ext_stimuli
    
    # The drive d always increases, limited to 100
    if drive < 100:  # we limit the drive value to 100
        drive += increment
        if drive > 100:
            drive = 100
    return motivation, drive # Return motivation and updated drive


# Returns the reward obtained when transiting to a new state: obstacle -10, other variation of drive values
# NOTE: The reward is now based on the VARIATION of ALL drives (energy + socialization)
def get_reward(new_state, drives_variation):
    if new_state in OBSTACLES:
        return -10
    else:
        # drives_variation is drives_before - drives_after (sum of all drives).
        # A positive variation means the total deficit decreased -> good reward.
        return drives_variation


def print_q_table(q_table):
    """
    Prints the Q-table as a grid with the optimal policy.
    """
    policy_grid = np.full((GRID_SIZE, GRID_SIZE), '', dtype=str)

    for i in range(GRID_SIZE):  # going through the rows (component y of the cells)
        for j in range(GRID_SIZE):  # going through the columns (component x of the cells)
            state = (j, i) # Assuming state access is (x, y) for Q-table indices
            
            if state in OBSTACLES:
                policy_grid[i, j] = 'X'
            else:
                # np.argmax needs state indices: q_table[x, y]
                best_action = np.argmax(q_table[j, i])
                policy_grid[i, j] = ACTION_MAP[best_action]

    print("Optimal Policy Grid:")
    for row in policy_grid:
        print(" ".join(row))


def print_detailed_q_table(q_table):
    """
    Prints the Q-values of the Q-table.
    """
    action_labels = [ACTION_MAP[a] for a in range(NUM_ACTIONS)]
    print(f"q-values for state (x, y): Q({action_labels})")
    
    # Loop all the map states
    for i in range(GRID_SIZE):  # going through the rows (component y of the cells)
        for j in range(GRID_SIZE):  # going through the columns (component x of the cells)
            # Assuming state access is (x, y) = (j, i)
            print(f"q-values for state {(j, i)}: {q_table[j][i].tolist()}")


def print_environment():
    """
    Prints the environment as a grid with the optimal policy, now including People.
    """
    policy_grid = np.full((GRID_SIZE, GRID_SIZE), '', dtype=str)

    for i in range(GRID_SIZE):  # going through the rows (component y of the cells)
        for j in range(GRID_SIZE):  # going through the columns (component x of the cells)
            state = (j, i)
            if state in OBSTACLES:
                policy_grid[i, j] = u"\U0001F32A" # Rock/Obstacle

            elif state in CHARGER_LOCATIONS:
                policy_grid[i, j] = u"\U0001F50B" # Battery/Charger
            
            # EXTENSION 4: Display People
            elif state in PEOPLE_LOCATIONS:
                policy_grid[i, j] = u"\U0001F464" # Person
                
            else:
                policy_grid[i, j] = u"\U0001F006" # Empty tile

    print("Grid:")
    for row in policy_grid:
        print(" ".join(row))


def get_action_epsilon_greedy(q_table, state):
    """
    Gets an action implementing the EPSILON-greedy strategy.
    """
    if np.random.rand() < EPSILON:
        new_action = np.random.choice(NUM_ACTIONS)  # Explore (Now 6 actions)
    else:
        # Accessing Q-table using the state tuple (x, y)
        new_action = np.argmax(q_table[state])  # Exploit
    return new_action


# Returns the next state after executing action from the current state. All transitions are deterministic.
def execute_action(current_state, action):
    x, y = current_state
    new_x, new_y = x, y
    
    # Define the movement based on the action and boundary checks
    if action == 0 and y > 0:  # Up (decrease y)
        new_y -= 1
    elif action == 1 and y < GRID_SIZE - 1:  # Down (increase y)
        new_y += 1
    elif action == 2 and x > 0:  # Left (decrease x)
        new_x -= 1
    elif action == 3 and x < GRID_SIZE - 1:  # Right (increase x)
        new_x += 1
    
    elif action == 4: # Charge (no movement)
        if current_state in CHARGER_LOCATIONS:
            CHARGE_AMOUNT = 10.0
            drive_values['energy'] = max(0.0, drive_values['energy'] - CHARGE_AMOUNT)
        pass
    
    # EXTENSION 5: New action 'interact' (index 5)
    elif action == 5: # Interact (no movement)
        if current_state in PEOPLE_LOCATIONS:
            # Satiate the need of socialization: d_socialized = 0.0
            drive_values['socialization'] = 0.0
        pass

    # Check if the move hit an obstacle: (x, y) -> (new_x, new_y)
    if (new_x, new_y) in OBSTACLES:
        new_x, new_y = x, y # If obstacle, stay in current state (no effect)
        
    return new_x, new_y


# Returns the sum of all drive values representing the general deficit (Total Motivation)
def get_total_drives(drives_values=drive_values):
    return sum(drives_values.values())


def run():
    # Define maximum number of iterations
    num_iter = MAX_ITER

    # Get initial state
    state = START_STATE
    
    # We run the control loop MAX_ITER iterations
    for iter_count in range(num_iter):
        # On every loop we get the previous value of the total drives (Energy + Socialization)
        drives_before = get_total_drives(drive_values)

        # Update Motivations and Drives
        
        # 1. Energy/Recharge Motivation
        if state in CHARGER_LOCATIONS:
            ext_stimuli_recharge = 10
        else:
            ext_stimuli_recharge = 0
        motivation_intensities['recharge'], drive_values['energy'] = update_motivation_drive(
            drive_values['energy'], DRIVE_ENERGY_INCREMENT_RATE, ext_stimuli_recharge)
        
        # 2. Socialization Motivation (EXTENSION 6: Update the new motivation/drive)
        if state in PEOPLE_LOCATIONS:
            ext_stimuli_socialize = 10 # Stimulus is +10 when next to a person
        else:
            ext_stimuli_socialize = 0
        motivation_intensities['socialize'], drive_values['socialization'] = update_motivation_drive(
            drive_values['socialization'], DRIVE_SOCIALIZATION_INCREMENT_RATE, ext_stimuli_socialize)

        # Print info on screen
        # print(f"iter {iter_count}: M_recharge={motivation_intensities['recharge']} d_energy={drive_values['energy']} M_socialize={motivation_intensities['socialize']} d_socialization={drive_values['socialization']}")

        # Select action to execute
        action = get_action_epsilon_greedy(q_table, state)

        # Execute action and observe next state
        next_state = execute_action(state, action)

        # Obtain reward
        drives_after = get_total_drives(drive_values)
        reward = get_reward(next_state, drives_before - drives_after) # Reward is based on total drive change

        # Update Q-value
        old_q_value = q_table[state][action]
        q_table[state][action] = (1 - LEARNING_RATE) * q_table[state][action] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(q_table[next_state]))
        state = next_state


# The robot will now prioritize actions that reduce the TOTAL drive deficit (energy or socialization)
run()
print("\n--- SOLUTION FOR EXERCISE 6: ENERGY AND SOCIALIZATION POLICY ---")
print_detailed_q_table(q_table)
print_q_table(q_table)
print_environment()