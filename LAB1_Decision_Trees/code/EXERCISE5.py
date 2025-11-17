import py_trees
class Failure(py_trees.behaviour.Behaviour):
    def __init__(self, name="Fail behavior"):
        super(Failure, self).__init__(name)
    def update(self):
        print("Returning FAILURE, but Inverter will flip it")
        return py_trees.common.Status.RUNNING
# Create an inverter that wraps a node that always fails
succeed_after_fail = py_trees.decorators.Inverter(name="Succeed after fail", child=Failure("Always Fail"))
# Create, run and display the behavior tree
bt = py_trees.trees.BehaviourTree(succeed_after_fail)
bt.tick()
print(py_trees.display.unicode_tree(root=bt.root, show_status=True))