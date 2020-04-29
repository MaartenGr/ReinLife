import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List
from TheGame.World.entities import Agent
from TheGame.Models.utils import BasicBrain


class Tracker:
    """ Used for tracking results of an experiment

    Variables tracked:
    * Avg Population Size
    * Avg Population Age
    * Avg Population Fitness
    * Best Population Age
    * Avg Number of Attacks
    * Avg Number of Kills
    * Avg Number of Intra Kills
    * Avg Number of Populations

    Parameters:
    -----------
    print_interval : int
        The interval at which to average the results and update the visualization

    interactive : bool, default False
        Whether to show an interactive matplotlib visualization

    google_colab : bool, default False
        Whether google colaboratory is used to execute the experiment

    nr_genes : int, default None
        The number of genes in the gene pool.
        NOTE: This is relevant if static families are used.

    static_families : bool, default True
        Whether the experiment is using static families

    brains : List[BasicBrain]
        All brains that are currently in the environment
    """
    def __init__(self,
                 print_interval: int,
                 interactive: bool = False,
                 google_colab: bool = False,
                 nr_genes: int = None,
                 static_families: bool = True,
                 brains: List[BasicBrain] = None):
        self.nr_genes = nr_genes

        # Tracks results each step
        self.track_results = {
            "Avg Population Size": {gene: [] for gene in range(nr_genes)},
            "Avg Population Age": {gene: [] for gene in range(nr_genes)},
            "Avg Population Fitness": {gene: [] for gene in range(nr_genes)},
            "Best Population Age": {gene: [] for gene in range(nr_genes)},
            "Avg Number of Attacks": {gene: [] for gene in range(nr_genes)},
            "Avg Number of Kills": {gene: [] for gene in range(nr_genes)},
            "Avg Number of Intra Kills": {gene: [] for gene in range(nr_genes)},
            "Avg Number of Populations": []
        }

        # Averages results from the tracker above each print_interval
        self.results = {
            "Avg Population Size": {gene: [] for gene in range(nr_genes)},
            "Avg Population Age": {gene: [] for gene in range(nr_genes)},
            "Avg Population Fitness": {gene: [] for gene in range(nr_genes)},
            "Best Population Age": {gene: [] for gene in range(nr_genes)},
            "Avg Number of Attacks": {gene: [] for gene in range(nr_genes)},
            "Avg Number of Kills": {gene: [] for gene in range(nr_genes)},
            "Avg Number of Intra Kills": {gene: [] for gene in range(nr_genes)},
            "Avg Number of Populations": []
        }

        self.variables = list(self.results.keys())
        self.print_interval = print_interval
        self.interactive = interactive
        self.google_colab = google_colab
        self.families = static_families
        self.colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
        self.first_run = True

        if not self.families:
            self.nr_genes = 1

        if interactive:
            columns = 3
            self.variables_to_plot = [self.get_val(self.variables, i, columns) for i in range(0, columns ** 2, columns)]

            # Prepare plot for google colab
            if self.google_colab:
                from google.colab import widgets
                self.grid = widgets.Grid(columns, columns)

            # Prepare plot for matplotlib
            else:
                self._init_matplotlib(columns, brains)
        else:
            self.fig = None

    def update_results(self, agents: List[Agent], n_epi: int):
        """ Update the results by averaging agent-specific values

        Parameters:
        -----------
        agents: List[Agent]
            All agents in the environment at n_epi

        n_epi : int
            The current episode number
        """
        self._track_results(agents)

        if n_epi % self.print_interval == 0 and n_epi != 0:
            self._average_results()

            if self.interactive:

                if self.google_colab:
                    self._plot_google()
                else:
                    self._plot_matplotlib()

    def _average_results(self):
        """ Average all results every print_interval """
        for variable in self.results.keys():
            if variable == "Avg Number of Populations":
                aggregation = self._aggregate(self.track_results["Avg Number of Populations"])
                self.results["Avg Number of Populations"].append(aggregation)

            else:
                for gen in range(self.nr_genes):
                    aggregation = self._aggregate(self.track_results[variable][gen])
                    self.results[variable][gen].append(aggregation)

    def _track_results(self, agents: List[Agent]):
        """ Update the results each step """
        if agents:
            self._track_avg_population_size(agents)
            self._track_avg_population_age(agents)
            self._track_avg_population_fitness(agents)
            self._track_best_age(agents)
            self._track_avg_nr_attacks(agents)
            self._track_avg_nr_kills(agents)
            self._track_avg_nr_populations(agents)

        # Append -1 to indicate that no results could be added
        else:
            for variable in self.track_results.keys():
                if variable != "Avg Number of Populations":
                    for gen in range(self.nr_genes):
                        self.track_results[variable][gen].append(-1)
                else:
                    self.track_results[variable].append(-1)

    def _track_avg_population_size(self, agents: List[Agent]):
        """ Track the average size of all populations """
        for gene in range(self.nr_genes):
            genes = [agent.gene for agent in agents if (agent.gene == gene and self.families) or (not self.families)]
            if len(genes) == 0:
                self.track_results["Avg Population Size"][gene].append(-1)
            else:
                unique, counts = np.unique(genes, return_counts=True)
                self.track_results["Avg Population Size"][gene].append(np.mean(counts))

    def _track_avg_population_age(self, agents: List[Agent]):
        """ Track the average age of all entities """
        for gene in range(self.nr_genes):
            pop_age = [agent.age for agent in agents if (agent.gene == gene and self.families) or (not self.families)]
            if len(pop_age) == 0:
                self.track_results["Avg Population Age"][gene].append(-1)
            else:
                self.track_results["Avg Population Age"][gene].append(np.mean(pop_age))

    def _track_avg_population_fitness(self, agents: List[Agent]):
        """ Update the average fitness/reward of all entities """
        for gene in range(self.nr_genes):
            fitness = [agent.reward for agent in agents if (agent.gene == gene and self.families) or
                       (not self.families)]
            if len(fitness) == 0:
                self.track_results["Avg Population Fitness"][gene].append(-1)
            else:
                self.track_results["Avg Population Fitness"][gene].append(np.mean(fitness))

    def _track_best_age(self, agents: List[Agent]):
        """ Track the highest age per gen """
        for gene in range(self.nr_genes):
            ages = [agent.age for agent in agents if (agent.gene == gene and self.families) or (not self.families)]
            if len(ages) == 0:
                self.track_results["Best Population Age"][gene].append(-1)
            else:
                max_age = max([agent.age for agent in agents if (agent.gene == gene and self.families) or
                                                         (not self.families)])
                self.track_results["Best Population Age"][gene].append(max_age)

    def _track_avg_nr_attacks(self, agents: List[Agent]):
        """ Track the average number of attacks per gen """
        for gene in range(self.nr_genes):
            actions = [agent.action for agent in agents if (agent.gene == gene and self.families) or
                       (not self.families)]
            attacks = [action for action in actions if action >= 4]
            if len(actions) == 0:
                self.track_results["Avg Number of Attacks"][gene].append(-1)
            else:
                self.track_results["Avg Number of Attacks"][gene].append(len(attacks) / len(actions))

    def _track_avg_nr_kills(self, agents: List[Agent]):
        """ Track the average number of kills per gen """
        for gene in range(self.nr_genes):
            killed = sum([agent.killed for agent in agents if (agent.gene == gene and self.families) or
                          (not self.families)])
            intra_killed = sum([agent.killed for agent in agents if (agent.gene == gene and self.families) or
                                (not self.families)])
            self.track_results["Avg Number of Kills"][gene].append(killed)

            if killed != 0:
                self.track_results["Avg Number of Intra Kills"][gene].append(intra_killed / killed)
            else:
                self.track_results["Avg Number of Intra Kills"][gene].append(0)

    def _track_avg_nr_populations(self, agents: List[Agent]):
        """ Track the average nr of populations  """
        genes = [agent.gene for agent in agents]
        self.track_results["Avg Number of Populations"].append(len(set(genes)))

    @staticmethod
    def get_val(data: List, i: int, col: int) -> List[bool]:
        """ Used to convert a 1d list to a 2d matrix even if their total sizes do not match """
        length = len(data[i:i + col])
        if length == col:
            return data[i:i + col]
        elif length > 0:
            return data[i:i + col] + [False for _ in range(col - length)]
        else:
            return [False for _ in range(col)]

    def _aggregate(self, a_list: List) -> np.array:
        aggregation = np.mean([val for val in a_list[-self.print_interval:] if val > -1])
        del a_list[:]
        return aggregation

    def _init_matplotlib(self, columns: int, brains: List[BasicBrain]):
        """ Initialize the matplotlib figure to draw on """
        plt.ion()
        self.fig, self.ax = plt.subplots(columns, columns, figsize=(14, 10))

        for i, _ in enumerate(self.variables_to_plot):
            for k, variable in enumerate(self.variables_to_plot[i]):
                if variable:

                    # Basic settings
                    self.ax[i][k].set_title(variable, fontsize=11)
                    self.ax[i][k].set_xlabel('Nr Episodes', fontsize=8)
                    self.ax[i][k].tick_params(axis='both', which='major', labelsize=7)

                    # Dark thick border
                    self.ax[i][k].spines['bottom'].set_linewidth('1.5')
                    self.ax[i][k].spines['left'].set_linewidth('1.5')

                    # Light border
                    self.ax[i][k].spines['top'].set_linewidth('1.5')
                    self.ax[i][k].spines['top'].set_color('#EEEEEE')
                    self.ax[i][k].spines['right'].set_linewidth('1.5')
                    self.ax[i][k].spines['right'].set_color('#EEEEEE')

                    # Grids
                    self.ax[i][k].grid(which='major', color='#EEEEEE', linestyle='-', linewidth=.5)
                    self.ax[i][k].minorticks_on()
                    self.ax[i][k].grid(which='minor', color='#EEEEEE', linestyle='--', linewidth=.5)

                else:
                    if self.families:
                        for gene in range(self.nr_genes):
                            label = f"{brains[gene].method}: Gen {gene}"
                            self.ax[i][k].plot([], [], label=label, color=self.colors[gene])
                    else:
                        self.ax[i][k].plot([], [], label="", color=self.colors[0])

                    self.ax[i][k].set_title("Legend", fontsize=12)
                    self.ax[i][k].legend(loc='upper center', frameon=False)
                    self.ax[i][k].axis('off')

        self.fig.tight_layout(pad=3.0)
        plt.show()

    def _plot_matplotlib(self):
        """ Plots intermediate results in matplotlib interactively

        NOTE: If you use pycharm, make sure to uncheck "show plots in tool window" in
        settings -> tools.
        """
        x = np.arange(self.print_interval,
                      (len(self.results["Avg Number of Populations"]) * self.print_interval) + 1,
                      self.print_interval)

        for i, _ in enumerate(self.variables_to_plot):
            for k, variable in enumerate(self.variables_to_plot[i]):
                if variable:
                    if type(self.results[variable]) == dict:

                        # Remove previous lines
                        if not self.first_run:
                            for gen in range(self.nr_genes):
                                self.ax[i][k].lines[0].remove()

                        # Plot new data
                        for gen in range(self.nr_genes):
                            self.ax[i][k].plot(x, self.results[variable][gen], label=gen, color=self.colors[gen])

                    else:
                        # Plot new data
                        if not self.first_run:
                            self.ax[i][k].lines[0].remove()
                        self.ax[i][k].plot(x, self.results[variable], color="#757575")

        if self.first_run:
            self.first_run = False

        plt.draw()
        mypause(0.0001)

    def _plot_google(self):
        """ Plots intermediate results in google colab """
        x = np.arange(self.print_interval, (len(self.results["Avg Number of Populations"]) * self.print_interval) + 1,
                      self.print_interval)

        for i, _ in enumerate(self.variables_to_plot):
            for k, variable in enumerate(self.variables_to_plot[i]):
                if variable:
                    if type(self.results[variable]) == dict:

                        # Plot new data
                        with self.grid.output_to(i, k):
                            self.grid.clear_cell()
                            plt.figure(figsize=(3, 3))
                            plt.title(variable)

                            for gen in range(self.nr_genes):
                                plt.plot(x, self.results[variable][gen], label=str(gen))
                            plt.legend()
                    else:

                        # Plot new data
                        with self.grid.output_to(i, k):
                            self.grid.clear_cell()
                            plt.figure(figsize=(3, 3))
                            plt.title(variable)
                            plt.plot(x, self.results[variable])


def mypause(interval: float):
    """ Needed to keep visualization minimized:

    https://stackoverflow.com/questions/45729092/make-interactive-matplotlib
    -window-not-pop-to-front-on-each-update-windows-7

    """
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return
