import py_trees
import random
class CheckLocation(py_trees.behaviour.Behaviour):
    def __init__(self, name,):
        super(CheckLocation, self).__init__(name)
    def update(self):
        at_location = random.choices([True, False], weights=[0.4, 0.6])[0]
        print(f"[Location Check] Robot in desired location? -> {at_location}")
        if at_location: return py_trees.common.Status.SUCCESS
        else: return py_trees.common.Status.FAILURE
class MoveToLocation(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(MoveToLocation, self).__init__(name)
    def update(self):
        success = random.choices([True, False], weights = [0.7, 0.3])[0]
        print(f"[Move To Location] Moved to desired location? -> {success}")
        if success:return py_trees.common.Status.SUCCESS 
        else: return py_trees.common.Status.FAILURE
class PickObject(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(PickObject, self).__init__(name)
    def update(self):
        success = random.choices([True, False], weights=[0.9, 0.1])[0]
        print(f"[Pick Object] Object picked? -> {success}")
        if success:
            print("ROBOT COMPLETED TASK")
            return py_trees.common.Status.SUCCESS 
        else:
            print("ROBOT NOT COMPLETED TASK")
            return py_trees.common.Status.FAILURE
def create_behavior_tree():

    already_there = py_trees.composites.Sequence("AlreadyThere",True,[CheckLocation("Check Location"), PickObject("Pick Object")])


    not_there = py_trees.composites.Sequence("NotThere",True,[MoveToLocation("Move To Location"), PickObject("Pick Object")])

    root = py_trees.composites.Selector("RootSelector",False,[already_there, not_there])

    return root


if __name__ == "__main__":
    root = create_behavior_tree()
    behaviour_tree = py_trees.trees.BehaviourTree(root)
    behaviour_tree.tick()
    py_trees.display.unicode_tree(behaviour_tree.root,show_status=True)