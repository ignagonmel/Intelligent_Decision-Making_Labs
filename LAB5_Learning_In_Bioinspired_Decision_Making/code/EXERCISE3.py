import numpy as np
import random 

# Initialize the environment
GRID_SIZE = 5

OBSTACLES = [(2, 2)]

START_STATE = (0, 0)
ACTION_MAP = {0: '↑', 1: '↓', 2: '←', 3: '→', 4: '🔌', 5: '🤝'}  
NUM_ACTIONS = len(ACTION_MAP) 

MAX_ITER = 100000 

# Q-learning parameters
LEARNING_RATE = 0.3
DISCOUNT_FACTOR = 0.7

# Epsilon parameter in the epsilon-greedy algorithm for action selection
EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DISCOUNT_RATE = 0.9999

# Defining motivations and drives
MOTIVATIONS_MAP = {'recharge': 0, 'socialize': 1}
DRIVES_MAP = {'energy': 0, 'socialization': 1}
DRIVE_ENERGY_INCREMENT_RATE = 2
DRIVE_SOCIALIZATION_INCREMENT_RATE = 1 
motivation_intensities = {'recharge': 0.0, 'socialize': 0.0} 
drive_values = {'energy': 0.0, 'socialization': 0.0} 

# Locations of external stimuli
CHARGER_LOCATIONS = [(0, 3)]
PEOPLE_LOCATIONS = [(3, 4), (4, 0)]

Q_TABLES = {
    'recharge': np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS)), 
    'socialize': np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS)) 
}


def update_motivation_drive(drive, increment, ext_stimuli=0):
    # Motivation = Drive + Stimuli 
    motivation = drive + ext_stimuli
    
    if drive < 100: 
        drive += increment
        if drive > 100:
            drive = 100
            
    return motivation, drive


def get_reward(new_state, drive_name, drive_before, drive_after, multiplier):
    if new_state in OBSTACLES:
        return -10
    
    drive_variation = drive_before - drive_after
    
    if drive_variation > 0:
        return drive_variation * multiplier
         
    return 0.0


def get_dominant_motivation(motivations_values=motivation_intensities):
    """Returns the key of the motivation with the highest intensity value."""
    dominant_key = max(motivations_values, key=motivations_values.get)
    return dominant_key


def print_q_table(Q_TABLES):
    """
    Prints the Optimal Policy Grid for all motivations, showing the pure argmax 
    of the Q-table without explicit action masking (as requested).
    """
    
    for motivation, q_table in Q_TABLES.items():
        policy_grid = np.full((GRID_SIZE, GRID_SIZE), '', dtype=str)

        for i in range(GRID_SIZE):  # rows (y)
            for j in range(GRID_SIZE):  # columns (x)
                state = (j, i) 
                
                if state in OBSTACLES:
                    policy_grid[i, j] = 'X'
                else:
                    # Se usa el argmax puro
                    best_action = np.argmax(q_table[j, i])
                    policy_grid[i, j] = ACTION_MAP[best_action]

        print(f"\nOptimal Policy Grid (Motivation: {motivation.upper()}):")
        for row in policy_grid:
            print(" ".join(row))


def print_detailed_q_table(Q_TABLES):
    """Prints the Q-values for all motivation Q-tables."""
    print("\n--- Detailed Q-tables per Motivation ---")
    action_labels = [ACTION_MAP[a] for a in range(NUM_ACTIONS)]
    
    for motivation, q_table in Q_TABLES.items():
        print(f"\n--- Q-values for DOMINANT MOTIVATION: {motivation.upper()} ---")
        print(f"q-values for state (x, y): Q({action_labels})")
        
        for i in range(GRID_SIZE):  
            for j in range(GRID_SIZE):  
                print(f"q-values for state {(j, i)}: {q_table[j][i].tolist()}")


def print_environment():
    """Prints the environment as a grid."""
    policy_grid = np.full((GRID_SIZE, GRID_SIZE), '', dtype=str)

    for i in range(GRID_SIZE):  
        for j in range(GRID_SIZE):  
            state = (j, i)
            if state in OBSTACLES:
                policy_grid[i, j] = u"\U0001F32A" # Rock/Obstacle
            elif state in CHARGER_LOCATIONS:
                policy_grid[i, j] = u"\U0001F50B" # Battery/Charger
            elif state in PEOPLE_LOCATIONS:
                policy_grid[i, j] = u"\U0001F464" # Person
            else:
                policy_grid[i, j] = u"\U0001F006" # Empty tile

    print("\nGrid:")
    for row in policy_grid:
        print(" ".join(row))


def get_action_epsilon_greedy(q_table, state):
    """
    Gets an action implementing the EPSILON-greedy strategy using the provided Q-table.
    """
    global EPSILON
    
    if np.random.rand() < EPSILON:
        new_action = np.random.choice(NUM_ACTIONS)  # Explore
    else:
        new_action = np.argmax(q_table[state]) # Exploit
        
    return new_action


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
    
    elif action == 5: # Interact (no movement)
        if current_state in PEOPLE_LOCATIONS:
            drive_values['socialization'] = 0.0
        pass

    # Check if the move hit an obstacle: (x, y) -> (new_x, new_y)
    if (new_x, new_y) in OBSTACLES:
        new_x, new_y = x, y # If obstacle, stay in current state (no effect)
        
    return new_x, new_y


def get_total_drives(drives_values=drive_values):
    return sum(drives_values.values())


def run():
    # Declarar globales para modificarlas
    global EPSILON, MIN_EPSILON, EPSILON_DISCOUNT_RATE
    
    num_iter = MAX_ITER
    state = START_STATE
    
    for iter_count in range(num_iter):

        # 1. Store drives BEFORE update (for specialized reward calculation if action is not successful)
        drive_energy_act_before = drive_values['energy']
        drive_socialization_act_before = drive_values['socialization']
        
        # 2. Update Motivations and Drives (Drive increase happens here)
        ext_stimuli_recharge = 10 if state in CHARGER_LOCATIONS else 0
        motivation_intensities['recharge'], drive_values['energy'] = update_motivation_drive(
            drive_values['energy'], DRIVE_ENERGY_INCREMENT_RATE, ext_stimuli_recharge)
        
        ext_stimuli_socialize = 10 if state in PEOPLE_LOCATIONS else 0
        motivation_intensities['socialize'], drive_values['socialization'] = update_motivation_drive(
            drive_values['socialization'], DRIVE_SOCIALIZATION_INCREMENT_RATE, ext_stimuli_socialize)

        # 3. Determine the DOMINANT MOTIVATION
        dominant_motivation = get_dominant_motivation()
        current_q_table = Q_TABLES[dominant_motivation]

        # 4. Select action
        action = get_action_epsilon_greedy(current_q_table, state)
        
        # 5. Execute action and observe next state
        next_state = execute_action(state, action) # drive_values are modified here if charge/interact

        # 6. Obtain Reward based on the DOMINANT DRIVE (using values from step 1 and values after step 5)
        drive_energy_act_after = drive_values['energy']
        drive_socialization_act_after = drive_values['socialization']

        if dominant_motivation == 'recharge':
            # Multiplier 10.0: Max reduction is 10 -> Max reward is ~100
            reward = get_reward(next_state, 'energy', drive_energy_act_before, drive_energy_act_after, multiplier=10.0)
        elif dominant_motivation == 'socialize':
            # Multiplier 1.0: Max reduction is ~100 -> Max reward is ~100
            reward = get_reward(next_state, 'socialization', drive_socialization_act_before, drive_socialization_act_after, multiplier=1.0)


        # 7. Update Q-value (max_next_q sin penalización)
        max_next_q = np.max(current_q_table[next_state])
        
        # 8. Actualización de la Q-table del motivador dominante
        current_q_table[state][action] = (1 - LEARNING_RATE) * current_q_table[state][action] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q)
        
        state = next_state
        
        # 9. Epsilon Decay
        EPSILON = EPSILON * EPSILON_DISCOUNT_RATE
        EPSILON = max(MIN_EPSILON, EPSILON)


run()
print("\n--- SOLUTION FOR EXERCISE 7: DOMINANT MOTIVATION LEARNING (FINAL) ---")
print_detailed_q_table(Q_TABLES)
print_q_table(Q_TABLES)
print_environment()