import py_trees
import random
class PrintOnce(py_trees.behaviour.Behaviour):
    def __init__(self, name="Print once"):
        super(PrintOnce, self).__init__(name)
    def update(self):
        if random.choice([True, False]):
            print("Message printed only once: SUCCESS")
            return py_trees.common.Status.SUCCESS
        else:
            print("Message printed only once: FAILURE")
            return py_trees.common.Status.FAILURE
# Use OneShot to ensure that PrintOnce runs only once
one_shot_print = py_trees.decorators.OneShot("One shot print", child=PrintOnce("Always Fail"), policy=py_trees.common.OneShotPolicy.ON_COMPLETION)
# Create, run and display the behavior tree
bt = py_trees.trees.BehaviourTree(one_shot_print)
bt.tick_tock(500)