import py_trees
import random
# Condition Node: Check for obstacles
class CheckForObstacles(py_trees.behaviour.Behaviour):
    def __init__(self, name="Check For Obstacles"):
        super(CheckForObstacles, self).__init__(name)
    def update(self):
 # Simulating detecting obstacle
        obstacle_detected = random.choices([True, False], weights=[0.25, 0.75])[0]
        if obstacle_detected:
            print("Obstacle detected!")
            return py_trees.common.Status.FAILURE
        else:
            print("No obstacle detected.")
            return py_trees.common.Status.SUCCESS
# Condition Node: Check battery status
class CheckBatteryLevel(py_trees.behaviour.Behaviour):
    def __init__(self, name="Check Battery Level", threshold=50):
        super(CheckBatteryLevel, self).__init__(name)
        self.threshold = threshold
    def update(self):
 # Simulating battery level between 1% and 100%
        battery_level = random.randint(1,100)
        if battery_level>self.threshold:
            print(f"Battery level ok: {battery_level}%")
            return py_trees.common.Status.SUCCESS
        else:
            print(f"Battery level too low: {battery_level}%")
            return py_trees.common.Status.FAILURE
# Action Node: Navigate
class Navigate(py_trees.behaviour.Behaviour):
    def __init__(self, name="Move Forward"):
        super(Navigate, self).__init__(name)
    def update(self):
        print("Navigating safely...")
        return py_trees.common.Status.SUCCESS
# Build the behavior tree
def create_behavior_tree():
    root = py_trees.composites.Sequence("Sequence: Check Then Navigate", memory=False)
    
    # Parallel Node: Run both checks simultaneously
    parallel_checks = py_trees.composites.Parallel(
        "Parallel: Obstacle and Battery Check",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(False)
    )
    
    # Obstacle and battery check nodes
    check_obstacles = CheckForObstacles()
    check_battery = CheckBatteryLevel()
    
    # Add both checks to parallel node
    parallel_checks.add_children([check_obstacles, check_battery])
    
    # Move forward action
    move_forward = Navigate()
    
    # Add parallel checks and navigation to sequence
    root.add_children([parallel_checks, move_forward])
    return root
# Main function to run the behavior tree
if __name__ == "__main__":
    root = create_behavior_tree()
    # Create and run the behavior tree
    bt = py_trees.trees.BehaviourTree(root)
    bt.tick()
