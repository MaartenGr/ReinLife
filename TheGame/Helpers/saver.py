import os
import json
import inspect
from datetime import date
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

from TheGame.World.entities import Agent


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

    Parameters:
    -----------
    main_folder : str
        The folder you would want to store the experiment.
        NOTE: This is just the name of the top folder. The exact location of the main_folder is determined
        by your curren working directory.

    google_colab : bool, default False
        Whether you use google colaboratory to run the experiment
    """
    def __init__(self, main_folder: str, google_colab: bool = False):
        cwd = os.getcwd()
        self.google_colab = google_colab
        self.separator = "/" if self.google_colab else "\\"
        self.main_folder = cwd + self.separator + main_folder

    def save(self, agents: List[Agent], family: bool, results: dict, settings: dict, fig: plt.Figure):
        """ Save brains and create directories if neccesary

        Parameters:
        -----------
        agents : List[Agent]
            A list of all agents for which paths need to be created

        family : bool
            Whether there are static families

        results : dict
            The results of the experiment

        settings : dict
            The settings of the experiment

        fig : plt.Figure
            The matplotlib figure to save
        """
        directory_paths, agent_paths, experiment_path = self._get_paths(agents, family)
        self._create_directories(directory_paths)

        for agent in agents:
            agent.save_brain(agent_paths[agent])

        with open(experiment_path + self.separator + "results.json", "w") as f:
            json.dump(results, f, indent=4)

        with open(experiment_path + self.separator + "settings.json", "w") as f:
            json.dump(settings, f, indent=4)

        if fig:
            fig.savefig(experiment_path + self.separator + "results.png", dpi=300)

        self._save_params(agents, agent_paths)

        print("################")
        print("Save Successful!")
        print("################")

    def _get_paths(self, agents: List[Agent], family: bool) -> Tuple[List[str], dict, str]:
        """ Get all paths for creating directories and paths for agents' brains

        Parameters:
        -----------
        agents : List[Agent]
            A list of all agents for which paths need to be created

        family : bool
            Whether there are static families

        Returns:
        --------
        all_paths : List[str]
            All paths that need to be created

        agents_paths : dict
            For each agent, the path that needs to be created

        experiment_path : str
            The main experiment folder
        """
        # Get experiment folder path and increment if one already exists executed on the same day
        today = str(date.today())
        experiment_path = self.main_folder + self.separator + today + "_V1"
        if os.path.exists(experiment_path):
            paths = [path for path in os.listdir(self.main_folder) if today in path]
            index = str(max([self.get_int(path.split("V")[-1]) for path in paths]) + 1)
            experiment_path = experiment_path[:-1] + index

        # Get path for each model in the experiment directory
        model_paths = list(set([experiment_path + self.separator + agent.brain.method for agent in agents]))

        # Extract paths for each agent based on their gene value
        if family:
            agents_paths = {agent: experiment_path + self.separator + agent.brain.method +
                            self.separator + "brain_gene_" + str(agent.gene)
                            for agent in agents}

        # If agents have the same model (i.e., "duplicates"), then increment their directory number
        else:
            agents_paths = {agent: experiment_path + self.separator + agent.brain.method + self.separator + "brain_1"
                            for agent in agents}

            vals, count = np.unique([val for val in agents_paths.values()], return_counts=True)
            duplicates = {x[0]: y[0] for x, y in zip(vals[np.argwhere(count > 1)], count[np.argwhere(count > 1)])}
            for duplicate in duplicates.keys():
                for count in range(duplicates[duplicate]):
                    agents_paths[self.get_key(duplicate, agents_paths)] = duplicate[:-1] + str(count+1)

        all_paths = [self.main_folder] + [experiment_path] + model_paths
        return all_paths, agents_paths, experiment_path

    def _create_directories(self, all_paths: List[str]):
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
    def _save_params(agents: List[Agent], agent_paths: dict):
        """ Extract and save the parameters for each brain

        Parameters:
        -----------
        agents : List[Agent]
            A list of all agents for which paths need to be created

        agents_paths : dict
            For each agent, the path that needs to be created
        """
        parameters = {agent: {} for agent in agents}

        # Extract parameters
        for agent in agents:
            params = inspect.getmembers(agent.brain, lambda a: not (inspect.isroutine(a)))
            for name, val in params:
                if type(val) in [float, int, bool, str]:
                    parameters[agent][name] = val

        # Save parameters
        for agent in agents:
            path = agent_paths[agent].replace("brain", "parameters") + ".json"
            with open(path, "w") as f:
                json.dump(parameters[agent], f, indent=4)

    @staticmethod
    def _create_directory(path: str) -> bool:
        """ Attempts to create a directory """
        try:
            os.mkdir(path)
        except OSError:
            return False
        else:
            return True

    @staticmethod
    def get_key(val: str, dictionary: dict) -> str:
        """ Gets the key of a value in a dictionary """
        return next(key for key, value in dictionary.items() if value == val)

    @staticmethod
    def get_int(a_string: str) -> int:
        """ Get all integers in a string """
        return int("".join([s for s in a_string if s.isdigit()]))
