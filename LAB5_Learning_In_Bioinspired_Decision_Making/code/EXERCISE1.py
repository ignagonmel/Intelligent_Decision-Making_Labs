import numpy as np

# Initialize the environment
grid_size = 5
goal_state = (4, 4)
obstacles = [(1, 2), (1, 1), (2, 1), (3, 1), (3, 3), (4, 3)]
#obstacles = [(2, 2)]

start_state = (0, 0)
action_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}

# Parameters
learning_rate = 0.1
discount_factor = 0.9
initial_epsilon = 1.0
epsilon = initial_epsilon
epsilon_discount_rate = 0.95
min_epsilon = 0.01
episodes = 100

# Q-table
q_table = np.zeros((grid_size, grid_size, 4))  # 4 actions: up, down, left, right


# states S = {s_0, s_1, ..., s_24}
#
#     x    0    1    2    3    4
# y     0 s_0  s_1  s_2  s_3  s_4
#       1 s_5  s_6  s_7  s_8  s_9
#       2 s_10 s_11 s_12 s_13 s_14
#       3 s_15 s_16 s_17 s_18 s_19
#       4 s_20 s_21 s_22 s_23 s_24


# Returns the next state considering the current state and an action. All transitions are deterministic.
def get_next_state(current_state, action):
    x, y = current_state

    new_x, new_y = x, y

    # Define the movement based on the action
    if action == 0:  # Up
        new_x -= 1
    elif action == 1:  # Down
        new_x += 1
    elif action == 2:  # Left
        new_y -= 1
    elif action == 3:  # Right
        new_y += 1

    # Check for wall collisions (stay in the same cell if hitting a wall) 
    if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
        return new_x, new_y

    else:
        # Wall collision: remain in the current state 
        return x, y
    

# Returns the reward obtained when transiting to a state: goal +10, obstacle -10, other -1
def get_reward(new_state):
    
    if new_state == goal_state:
        return 10
    elif new_state in obstacles:
        return -10
    else:
        return -1
def print_q_table(q_table, goal_state):
    """
    Prints the Q-table as a grid with the optimal policy.

    Parameters:
    - q_table: A numpy array of shape (grid_size, grid_size, 4) representing Q-values for actions.
    - goal_state: A tuple (x, y) marking the goal position.
    - obstacle_states: A tuple (x, y) marking the obstacle position.
    """
    policy_grid = np.full((grid_size, grid_size), '', dtype=str)

    for i in range(grid_size):  # going through the rows (component y of the cells)
        for j in range(grid_size):  # going through the columns (component x of the cells)
            if (i, j) == goal_state:
                policy_grid[i, j] = 'G'
            elif (i, j) in obstacles:
                policy_grid[i, j] = 'X'
            else:
                best_action = np.argmax(q_table[i, j])
                policy_grid[i, j] = action_map[best_action]

    print("Optimal Policy Grid:")
    for row in policy_grid:
        print(" ".join(row))


def print_detailed_q_table(q_table):
    """
    Prints the Q-values of the Q-table.

    Parameters:
    - q_table: A numpy array of shape (grid_size, grid_size, 4) representing Q-values for actions.
    """
    print(
        f"q-values for state (i, j): Q((i,j),{action_map[0]}), Q((i,j),{action_map[1]}), Q((i,j),{action_map[2]}), Q((i,j),{action_map[3]})")
    for i in range(grid_size):  # going through the rows (component y of the cells)
        for j in range(grid_size):  # going through the columns (component x of the cells)
            print(f"q-values for state {(i, j)}: {q_table[i][j].tolist()}")


def get_action_epsilon_greedy(q_table, state):
    """
    Gets an action implementing the epsilon-greedy strategy to balance exploration and exploitation:
    with a probability of 1−ϵ, the agent chooses the action with the highest Q-value for the current state;
    with a probability of ϵ, the agent selects a random action, regardless of its Q-value.

    Parameters:
    - q_table: A numpy array of shape (grid_size, grid_size, 4) representing Q-values for actions.
    - state: The current state of the agent
    """

    if np.random.rand() < epsilon:
        action_chosen = np.random.choice(4)
    else:
        action_chosen = np.argmax(q_table[state])
    
    return action_chosen

# Q-learning loop
for episode in range(episodes):
    state = start_state
    iter = 0
    while state != goal_state:
        iter += 1
        x, y = state
        action = get_action_epsilon_greedy(q_table, state)
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        next_x, next_y = next_state
        # For exercise 3:
        if next_state in obstacles: # if hitting an obstacle the robot remains in its cell
            next_state = state
            next_x, next_y = state
        max_future_q = np.max(q_table[next_x, next_y, :])
        current_q = q_table[x, y, action]
        q_table[state][action] = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
        print(f"episode {episode} iteration {iter}: Q(state {state}, action {action_map[action]})={current_q}, new Q({state}, {action_map[action]})={q_table[state][action]}, reward {reward}, new state {next_state}")
        state = next_state
    epsilon = max(min_epsilon, epsilon * epsilon_discount_rate)

print("Trained Q-table:")
# print(q_table)
print_detailed_q_table(q_table)
print_q_table(q_table, goal_state)
