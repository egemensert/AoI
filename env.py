class Environment:
    def __init__(self):
        pass

    def reset(self):
        """This method should reset the environment and return the current
        state (list of float(s))"""
        return state

    def step(self, action):
        """ Main function of the environment. The method
        simulates the environment according to the given action, that is an
        integer, defining the action matrix's index. The function should
        return the state (list of float(s)) after running according to the
        action, the reward (float), whether the simulation is done (boolean)
        and a dictionary that logs the system (optional)."""
        return state, reward, done, info
