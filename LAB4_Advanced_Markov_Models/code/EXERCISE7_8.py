"""The Searching Object Problem.

This is a POMDP problem; Namely, it specifies both
the POMDP (i.e. state, action, observation space)
and the T/O/R for the agent as well as the environment.

A robot in a warehouse is tasked with finding a specific object among four shelves. Each
shelf has to be inspected to find the object. The robot has a limited view and can only
observe the contents of a shelf when it is close enough. The robot’s objective is to locate
the desired object while minimizing unnecessary movement and avoiding miss-identifications.

The robot can:
•	Move Left or Move Right between shelves.
•	Inspect the current shelf to identify if it contains the object.

The robot receives a reward when it finds the target object and incurs a small penalty for
each move and a larger penalty for a miss-identification (searching for the object but not
finding it).

States: shelf1, shelf2, shelf3, shelf4
Actions: inspect, move-left, move-right
Rewards:
    +50 for finding the object. -10 for miss-identification.
    -1 for moving.
Observations: You can detect either "object-found", or "not-found", but consider that the
perceptual information is noisy.

"""

import pomdp_py
import random

NUM_SHELVES = 4 # number of states where the desired object might be

# Define the states
class ObjectState(pomdp_py.State):
    S0 = "shelf1"
    S1 = "shelf2"
    S2 = "shelf3"
    S3 = "shelf4"

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, ObjectState):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "ObjectState(%s)" % self.name

    def other(self):
        if self.name == ObjectState.S0:
            return ObjectState(random.choice([ObjectState.S1, ObjectState.S2, ObjectState.S3]))
        elif self.name == ObjectState.S1:
            return ObjectState(random.choice([ObjectState.S0, ObjectState.S2, ObjectState.S3]))
        elif self.name == ObjectState.S2:
            return ObjectState(random.choice([ObjectState.S0, ObjectState.S1, ObjectState.S3]))
        else:
            return ObjectState(random.choice([ObjectState.S0, ObjectState.S1, ObjectState.S2]))


OBJECT_STATE = ObjectState(ObjectState.S1) # Hidden position of the target object (unknown to the agent)

# Define the actions
class ObjectAction(pomdp_py.Action):
    INSPECT = "inspect"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, ObjectAction):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "ObjectAction(%s)" % self.name

# Define the robot's observations
class ObjectObservation(pomdp_py.Observation):
    FOUND_OBJECT = "found"
    NOT_FOUND = "not_found"

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, ObjectObservation):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "ObjectObservation(%s)" % self.name


# Define the observation model--> if the robot inspects the correct shelf, it has a high probability of observing that
#                                 the object is present; otherwise, it likely observes that the object is not there.
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
        self.noise = noise

    # The probability method returns the probability of receiving a particular observation, given next_state and action.
    def probability(self, observation, next_state, action):
        if action.name == ObjectAction.INSPECT:
            if observation.name == ObjectObservation.FOUND_OBJECT and next_state.name == OBJECT_STATE.name:
                return 1.0 - self.noise
            elif observation.name == ObjectObservation.NOT_FOUND and next_state.name == OBJECT_STATE.name:
                return self.noise
            elif observation.name == ObjectObservation.NOT_FOUND and next_state.name != OBJECT_STATE.name:
                return 1.0 - self.noise
            elif observation.name == ObjectObservation.FOUND_OBJECT and next_state.name != OBJECT_STATE.name:
                return self.noise
        else:
            if observation.name == ObjectObservation.NOT_FOUND:
                return 1.0
            elif observation.name == ObjectObservation.FOUND_OBJECT:
                return 0.0  
        

    # The sample method generates a random observation based on the current next_state and the action taken.
    # This method represents the probabilistic nature of observations, which are often noisy or limited in scope.
    def sample(self, next_state, action):
        thresh = 0
        if action.name == ObjectAction.INSPECT:
            # --- To Be Completed HERE ---
            if next_state.name == OBJECT_STATE.name:
                thresh = 1.0 - self.noise  # correct shelf
            else:
                thresh = self.noise  # false positive probability
        else:
            # For non-inspect actions → always NOT_FOUND
            return ObjectObservation(ObjectObservation.NOT_FOUND)

        if random.uniform(0, 1) < thresh:
            return ObjectObservation(ObjectObservation.FOUND_OBJECT)
        else:
            return ObjectObservation(ObjectObservation.NOT_FOUND)

    def get_all_observations(self):
        return [ObjectObservation(s) for s in {ObjectObservation.FOUND_OBJECT, ObjectObservation.NOT_FOUND}]


# Define the transition model
class TransitionModel(pomdp_py.TransitionModel):

    # The probability method returns the probability of reaching a next_state from a given state after taking an action
    def probability(self, next_state, state, action):
        if action.name == ObjectAction.INSPECT:
            if next_state.name == state.name:
                return 0.9  # 90% chance of staying in the same place
            else:
                return 0.1  # 10% chance of moving one position right or left
        else:
            # This is the first example: if action is move left, calculate the probability of ending in another state
            if action.name == ObjectAction.MOVE_LEFT:
                if state.name == ObjectState.S0:
                    if next_state.name == ObjectState.S0:
                        return 0.95  # 95% chance of staying in the same place
                    elif next_state.name == ObjectState.S1:
                        return 0.05  # 5% chance of moving to the right
                    else:
                        return 0
                 # From S1
                elif state.name == ObjectState.S1:
                    if next_state.name == ObjectState.S0:
                        return 0.95
                    elif next_state.name == ObjectState.S1:
                        return 0.04
                    elif next_state.name == ObjectState.S2:
                        return 0.01
                    else:
                        return 0

                # From S2
                elif state.name == ObjectState.S2:
                    if next_state.name == ObjectState.S1:
                        return 0.95
                    elif next_state.name == ObjectState.S2:
                        return 0.04
                    elif next_state.name == ObjectState.S3:
                        return 0.01
                    else:
                        return 0

                # From S3
                elif state.name == ObjectState.S3:
                    if next_state.name == ObjectState.S2:
                        return 0.95
                    elif next_state.name == ObjectState.S3:
                        return 0.05
                    else:
                        return 0

        # MOVE RIGHT BEHAVIOR
        if action.name == ObjectAction.MOVE_RIGHT:

            # From S0
            if state.name == ObjectState.S0:
                if next_state.name == ObjectState.S1:
                    return 0.95
                elif next_state.name == ObjectState.S0:
                    return 0.05
                else:
                    return 0

            # From S1
            elif state.name == ObjectState.S1:
                if next_state.name == ObjectState.S2:
                    return 0.95
                elif next_state.name == ObjectState.S1:
                    return 0.04
                elif next_state.name == ObjectState.S0:
                    return 0.01
                else:
                    return 0

            # From S2
            elif state.name == ObjectState.S2:
                if next_state.name == ObjectState.S3:
                    return 0.95
                elif next_state.name == ObjectState.S2:
                    return 0.04
                elif next_state.name == ObjectState.S1:
                    return 0.01
                else:
                    return 0

            # From S3 (cannot move right)
            elif state.name == ObjectState.S3:
                if next_state.name == ObjectState.S3:
                    return 0.95
                elif next_state.name == ObjectState.S2:
                    return 0.05
                else:
                    return 0

        # Default fallback
        return 0


    #  The sample method returns a sampled next state based on the current state and action
    def sample(self, state, action):
        next_state_name = state.name
        all_state_names = [ObjectState.S0, ObjectState.S1, ObjectState.S2, ObjectState.S3]
        if state.name == ObjectState.S0:
            if action.name in [ObjectAction.MOVE_LEFT, ObjectAction.INSPECT]:
                next_state_name = random.choices(all_state_names, [95, 5, 0, 0])[0]
            else:  # ObjectAction.MOVE_RIGHT
                next_state_name = random.choices(all_state_names, [5, 95, 0, 0])[0]
        elif state.name == ObjectState.S1:
            if action.name == ObjectAction.INSPECT:
                next_state_name = random.choices(all_state_names, [5, 90, 5, 0])[0]
            elif action.name == ObjectAction.MOVE_LEFT:
                next_state_name = random.choices(all_state_names, [95, 4, 1, 0])[0]
            else:  # ObjectAction.MOVE_RIGHT
                next_state_name = random.choices(all_state_names, [1, 4, 95, 0])[0]

        elif state.name == ObjectState.S2:
            if action.name == ObjectAction.INSPECT:
                next_state_name = random.choices(all_state_names, [0, 5, 90, 5])[0]
            elif action.name == ObjectAction.MOVE_LEFT:
                next_state_name = random.choices(all_state_names, [0, 1, 4, 95])[0]
            else:  # MOVE_RIGHT
                next_state_name = random.choices(all_state_names, [0, 0, 1, 99])[0]

        else:  # ObjectState.S3
            if action.name == ObjectAction.INSPECT:
                next_state_name = random.choices(all_state_names, [0, 0, 5, 95])[0]
            elif action.name == ObjectAction.MOVE_LEFT:
                next_state_name = random.choices(all_state_names, [0, 1, 4, 95])[0]
            else:  # MOVE_RIGHT at last shelf (should not move)
                next_state_name = random.choices(all_state_names, [0, 0, 5, 95])[0]

        return ObjectState(next_state_name)#

    def get_all_states(self):
        return [ObjectState(s) for s in {ObjectState.S0, ObjectState.S1, ObjectState.S2, ObjectState.S3}]


# Define the reward model
class RewardModel(pomdp_py.RewardModel):
    # The reward method is responsible for returning the reward associated with transitioning from a given state to a
    # next_state due to a particular action. This function is central to shaping the agent’s behavior by defining
    # which actions are desirable or undesirable in specific contexts.
    def _reward_func(self, state, action):
        # FOUND the object (correct shelf AND inspecting)
        if action.name == ObjectAction.INSPECT and state.name == OBJECT_STATE.name:
            return 50

        # Miss-identification (inspecting wrong shelf)
        elif action.name == ObjectAction.INSPECT and state.name != OBJECT_STATE.name:
            return -10

        # Movement cost
        else:
            return -1

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)


# Policy Model
class PolicyModel(pomdp_py.RolloutPolicy):
    """A simple policy model with uniform prior over a
    small, finite action space"""

    ACTIONS = [ObjectAction(s) for s in {ObjectAction.INSPECT, ObjectAction.MOVE_LEFT, ObjectAction.MOVE_RIGHT}]

    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def rollout(self, state, history=None):
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS


# Define the POMDP Problem
class ObjectProblem(pomdp_py.POMDP):
    def __init__(self, obs_noise, init_true_state, init_belief):

        # The agent has access to the belief (initially a probabilistic guess of the environment's state) and models
        # for transitions, observations, and rewards. agent encapsulates the agent's models and belief
        agent = pomdp_py.Agent(
            init_belief,                 # initial belief distribution over the possible state
            PolicyModel(),               # policy model used to choose actions
            TransitionModel(),           # transition model
            ObservationModel(obs_noise), # observation model
            RewardModel(),               # reward model
        )

        # env represents the true environment with access to the state, transition model, and reward model
        env = pomdp_py.Environment(
            init_true_state,
            TransitionModel(),
            RewardModel()
        )

        super().__init__(agent, env, name="ObjectProblem")


def test_planner(object_problem, planner, nsteps=3):
    """
    Runs the action-feedback loop of Object problem POMDP

    Args:
        object_problem (ObjectProblem): a problem instance
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
    """
    for i in range(nsteps):
        action = planner.plan(object_problem.agent)

        print("==== Step %d ====" % (i + 1))
        print(f"True state: {object_problem.env.state}")
        print(f"Belief: {object_problem.agent.cur_belief}")
        print(f"Action: {action}")

        reward = object_problem.env.state_transition(action, execute=True)  # the environment state is transitioned
        print("Reward:", reward)

        # Let's create some simulated real observation;
        real_observation = object_problem.agent.observation_model.sample(object_problem.env.state, action)
        print(">> Observation:", real_observation)
        object_problem.agent.update_history(action, real_observation)

        # Update the belief. If the planner is POMCP, planner.update
        # also automatically updates agent belief.
        planner.update(object_problem.agent, action, real_observation)
        if isinstance(object_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(
                object_problem.agent.cur_belief,
                action,
                real_observation,
                object_problem.agent.observation_model,
                object_problem.agent.transition_model,
            )
            object_problem.agent.set_belief(new_belief)


def make_object_problem(noise=0.15, init_state=ObjectState.S0, init_belief=[0.25, 0.25, 0.25, 0.25]):
    """Convenient function to quickly build a my_object_problem domain.
    Useful for testing"""
    my_object_problem = ObjectProblem(
        noise,
        ObjectState(init_state),
        pomdp_py.Histogram(
            {
                ObjectState(ObjectState.S0): init_belief[0],
                ObjectState(ObjectState.S1): init_belief[1],
                ObjectState(ObjectState.S2): init_belief[2],
                ObjectState(ObjectState.S3): init_belief[3]
            }
        ),
    )
    return my_object_problem


if __name__ == "__main__":
    search_object = make_object_problem()

    print("** Testing value iteration **")
    vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
    test_planner(search_object, vi, nsteps=10)
