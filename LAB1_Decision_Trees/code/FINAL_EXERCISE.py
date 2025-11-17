import py_trees
import random
import time



class DetectHuman(py_trees.behaviour.Behaviour):
    def __init__(self, name = "Detect Human"):
        super(DetectHuman, self).__init__(name)
    def update(self):
        print("Trying to detect human")
        if random.choice([True, False]):
            print("HUMAN DETECTED")
            return py_trees.common.Status.SUCCESS
        else:
            print("HUMAN NOT DETECTED")
            return py_trees.common.Status.FAILURE

class Greet(py_trees.behaviour.Behaviour):
    def __init__(self, name = "Greet"):
        super(Greet, self).__init__(name)
    def update(self):
        print("Greetings human")
        return py_trees.common.Status.SUCCESS

class EngageInConversation(py_trees.behaviour.Behaviour):
    def __init__(self, name = "Conversation"):
        super(EngageInConversation, self).__init__(name)
    def update(self):
        print("How are you human?")
        print("I am a social robot")
        print("My job is to provide you assistance")
        return py_trees.common.Status.SUCCESS

class NeedAssistance(py_trees.behaviour.Behaviour):
    def __init__(self, name = "Need Assistance"):
        super(NeedAssistance, self).__init__(name)
    def update(self):
        print("Do you need any assistance human?")
        if random.choice([True, False]):
            print("Perfect i will assist you")
            return py_trees.common.Status.SUCCESS
        else:
            print("Okay human, you do not need my help")
            return py_trees.common.Status.FAILURE

class ProvideAssistance(py_trees.behaviour.Behaviour):
    def __init__(self, name = "Provide Assistance"):
        super(ProvideAssistance, self).__init__(name)
    def update(self):
        print("ASSISTANCE PROVIDED")
        return py_trees.common.Status.SUCCESS

class Idle(py_trees.behaviour.Behaviour):
    def __init__(self, name = "Idle"):
        super(Idle, self).__init__(name)
    def update(self):
        print("IDLE STATE")
        time.sleep(2)
        return py_trees.common.Status.SUCCESS


def create_behavior_tree():
    HumanDetect = DetectHuman()
    HumanDetect3 = py_trees.decorators.Retry("Detect human 3 times", HumanDetect, num_failures=3)
    Greetings = Greet()
    GreetOneTime = py_trees.decorators.OneShot("One shot greet", Greetings, py_trees.common.OneShotPolicy.ON_COMPLETION)
    Conversation = EngageInConversation()
    AskAssistance = NeedAssistance()
    Help = ProvideAssistance()
    IdleState = Idle()
    root = py_trees.composites.Sequence("Root sequence", True, [HumanDetect3, GreetOneTime, Conversation, AskAssistance, Help, IdleState])
    return root

if __name__ == "__main__":
    SocialRobot = create_behavior_tree()
    bt = py_trees.trees.BehaviourTree(SocialRobot)
    bt.tick_tock(500)
    py_trees.display.unicode_tree(bt.root,show_status=True)