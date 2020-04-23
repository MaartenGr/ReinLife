import os
import json
from datetime import date
import numpy as np


class Saver:
    """ Class for saving the results of an experiment

    If directories do not currently exist, it will create them. The general structure
    of the experiment will be:

        .
        ├── 2020-04-22_V1
        │   └── PPO
        │   │   └── brain_1.pt
        │   │   └── brain_2.pt
        │   └── PERD3QN
        │   │   └── brain_1.pt
        │   └── DQN
        │   │   └── brain_1.pt
        │   └── Results...
        │
        ├── 2020-04-22_V1
        │   └── PPO
        │   │   └── brain_1.pt
        │   └── PERD3QN
        │       └── brain_1.pt

        etc.

    Thus, each experiment is defined by the date of saving the models and an additional "_V1" if multiple
    experiments were performed on the same day. Within each experiment, each model is saved into a
    directory of its model class, for example PPO, PERD3QN, and DQN. Then, each model is saved as
    "brain_x.pt" where x is simply the sequence in which it is saved.

    """
    def __init__(self, main_folder):
        cwd = os.getcwd()
        self.main_folder = cwd + "\\" + main_folder

    def save(self, agents, family, results):
        """ Save brains and create directories if neccesary """
        directory_paths, agent_paths, experiment_path = self._get_paths(agents)
        self._create_directories(directory_paths)

        if family:
            for agent in agents:
                agent.save_brain(agent_paths[agent] + "_gen_" + str(agent.gen))
        else:
            for agent in agents:
                agent.save_brain(agent_paths[agent])

        with open(experiment_path + "\\" + "results.json", "w") as f:
            json.dump(results, f, indent=4)

        print("################")
        print("Save Successful!")
        print("################")

    def _get_paths(self, agents):
        """ Get all paths for creating directories and paths for agents' brains """
        # Get experiment folder path and increment if one already exists executed on the same day
        today = str(date.today())
        experiment_path = self.main_folder + "\\" + today + "_V1"
        if os.path.exists(experiment_path):
            paths = [path for path in os.listdir(self.main_folder) if today in path]
            index = str(max([self.get_int(path.split("V")[-1]) for path in paths]) + 1)
            experiment_path = experiment_path[:-1] + index

        # Get path for each model in the experiment directory
        model_paths = list(set([experiment_path + "\\" + agent.brain.method for agent in agents]))

        # If agents have the same model (i.e., "duplicates"), then increment their directory number
        agents_paths = {agent: experiment_path + "\\" + agent.brain.method + "\\" + "brain_1" for agent in agents}
        vals, count = np.unique([val for val in agents_paths.values()], return_counts=True)
        duplicates = {x[0]: y[0] for x, y in zip(vals[np.argwhere(count > 1)], count[np.argwhere(count > 1)])}
        for duplicate in duplicates.keys():
            for count in range(duplicates[duplicate]):
                agents_paths[self.get_key(duplicate, agents_paths)] = duplicate[:-1] + str(count+1)

        all_paths = [self.main_folder] + [experiment_path] + model_paths
        return all_paths, agents_paths, experiment_path

    def _create_directories(self, all_paths):
        """ Create directories if neccesary and print which were created """
        created_paths = []

        for path in all_paths:
            if not os.path.exists(path):
                if self._create_directory(path):
                    created_paths.append(path)
                else:
                    raise Exception(f'{path} could not be created')

        if created_paths:
            print("The following directories were created: ")
            for path in created_paths:
                print(f"* {path}")
            print()

    @staticmethod
    def _create_directory(path):
        """ Tries to create a directory """
        try:
            os.mkdir(path)
        except OSError:
            return False
        else:
            return True

    @staticmethod
    def get_key(val, dictionary):
        """ Gets the key of a value in a dictiory """
        return next(key for key, value in dictionary.items() if value == val)

    @staticmethod
    def get_int(a_string):
        """ Get all integers in a string """
        return int("".join([s for s in a_string if s.isdigit()]))
