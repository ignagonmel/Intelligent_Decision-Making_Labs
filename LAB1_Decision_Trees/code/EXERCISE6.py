import py_trees
class AlwaysFail(py_trees.behaviour.Behaviour):
    def __init__(self, name="Always fail"):
        super(AlwaysFail, self).__init__(name)
    def update(self):
        print("Always failing")
        return py_trees.common.Status.FAILURE
# Retry the action 3 times before giving up
three_times_retry = py_trees.decorators.Retry("three times retry",child=AlwaysFail("Always Fail"), num_failures=3)
# Create, run and display the behavior tree
bt = py_trees.trees.BehaviourTree(three_times_retry)
for i in range(4):
    print(f"Tick {i+1}")
    bt.tick()
    print(py_trees.display.unicode_tree(root=bt.root, show_status=True))