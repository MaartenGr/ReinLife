import os
import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

GRID_COORD_MARGIN_SIZE = 0
CELL_SIZE = 16
BLACK = (255, 0, 0)


class Grid:
    def __init__(self, surface, cellSize):
        self.surface = surface
        self.colNb = surface.get_width() // cellSize
        self.lineNb = surface.get_height() // cellSize
        self.cellSize = cellSize
        self.grid = [[0 for i in range(self.colNb)] for j in range(self.lineNb)]
        self.font = pygame.font.SysFont('arial', 12, False)

    def create_grid(self):
        for li in range(self.lineNb + 1):
            liCoord = GRID_COORD_MARGIN_SIZE + li * CELL_SIZE
            pygame.draw.line(self.surface, BLACK, (GRID_COORD_MARGIN_SIZE, liCoord),
                             (self.surface.get_width(), liCoord))
        for co in range(self.colNb + 1):
            colCoord = GRID_COORD_MARGIN_SIZE + co * CELL_SIZE
            pygame.draw.line(self.surface, BLACK, (colCoord, GRID_COORD_MARGIN_SIZE),
                             (colCoord, self.surface.get_height()))


def check_pygame_exit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return False
    return True


class Results:
    def __init__(self, print_interval, interactive=False, save_visualization=False, google_colab=False,
                 nr_gens=None):

        # Average within and between episodes
        self.nr_gens = nr_gens
        self.avg_population_size = {gen: [] for gen in range(nr_gens)}
        self.avg_population_age = {gen: [] for gen in range(nr_gens)}
        self.avg_population_fitness = {gen: [] for gen in range(nr_gens)}
        self.track_avg_population_size = {gen: [] for gen in range(nr_gens)}
        self.track_avg_population_age = {gen: [] for gen in range(nr_gens)}
        self.track_avg_population_fitness = {gen: [] for gen in range(nr_gens)}

        # Average between episodes and max within episodes
        self.avg_best_size = {gen: [] for gen in range(nr_gens)}
        self.avg_best_age = {gen: [] for gen in range(nr_gens)}
        self.avg_best_fitness = {gen: [] for gen in range(nr_gens)}
        self.track_avg_best_size = {gen: [] for gen in range(nr_gens)}
        self.track_avg_best_age = {gen: [] for gen in range(nr_gens)}
        self.track_avg_best_fitness = {gen: [] for gen in range(nr_gens)}

        self.avg_nr_attacks = {gen: [] for gen in range(nr_gens)}
        self.track_avg_nr_attacks = {gen: [] for gen in range(nr_gens)}

        self.avg_nr_populations = []
        self.track_avg_nr_populations = []

        self.oldest_entities = []
        self.max_age_list = []
        self.best_fitness_list = []
        self.best_fitness = -1_000_000
        self.print_interval = print_interval
        self.interactive = interactive
        self.save_visualization = save_visualization
        self.google_colab = google_colab

        if interactive:

            if self.google_colab:
                from google.colab import widgets
                self.grid = widgets.Grid(3, 3)
            else:
                plt.ion()
                self.fig, self.ax = plt.subplots(3, 3, sharex=True)

                self.ax[0][0].set_title("Average Population Age", fontsize=8)
                self.ax[0][0].set_xlabel('', fontsize=8)
                self.ax[0][0].tick_params(axis='both', which='major', labelsize=6)

                self.ax[0][1].set_title("Average Population Fitness", fontsize=8)
                self.ax[0][1].set_xlabel('', fontsize=8)
                self.ax[0][1].tick_params(axis='both', which='major', labelsize=6)

                self.ax[0][2].set_title("Average Population Size", fontsize=8)
                self.ax[0][2].set_xlabel('', fontsize=8)
                self.ax[0][2].tick_params(axis='both', which='major', labelsize=6)

                self.ax[1][0].set_title("Best Population Age", fontsize=8)
                self.ax[1][0].tick_params(axis='both', which='major', labelsize=6)

                self.ax[1][1].set_title("Best Population Fitness", fontsize=8)
                self.ax[1][1].tick_params(axis='both', which='major', labelsize=6)

                self.ax[1][2].set_title("Best Population Size", fontsize=8)
                self.ax[1][2].tick_params(axis='both', which='major', labelsize=6)

                self.ax[2][0].set_title("Average nr Attacks", fontsize=8)
                self.ax[2][0].set_xlabel('Nr Episodes', fontsize=8)
                self.ax[2][0].tick_params(axis='both', which='major', labelsize=6)

                self.ax[2][1].set_title("XXX", fontsize=8)
                self.ax[2][1].set_xlabel('Nr Episodes', fontsize=8)
                self.ax[2][1].tick_params(axis='both', which='major', labelsize=6)

                self.ax[2][2].set_title("XXX", fontsize=8)
                self.ax[2][2].set_xlabel('Nr Episodes', fontsize=8)
                self.ax[2][2].tick_params(axis='both', which='major', labelsize=6)

                plt.show(block=False)

    def _update_avg_nr_attacks(self, agents):
        """ Track the average size of all populations """
        if agents:
            for gen in range(self.nr_gens):
                actions = [agent.action for agent in agents if agent.gen == gen]
                attacks = [action for action in actions if action >= 4]
                if len(actions) == 0:
                    self.track_avg_nr_attacks[gen].append(-1)
                else:
                    self.track_avg_nr_attacks[gen].append(len(attacks) / len(actions))
        else:
            for gen in range(self.nr_gens):
                self.track_avg_nr_attacks[gen].append(-1)

    def _update_avg_nr_populations(self, agents):
        """ Update the best fitness of all entities """
        """ Track the average size of all populations """
        if agents:
            gens = [agent.gen for agent in agents]
            self.track_avg_nr_populations.append(len(set(gens)))
        else:
            self.track_avg_nr_populations.append(-1)

    def _update_avg_population_size(self, agents):
        """ Track the average size of all populations """
        if agents:
            for gen in range(self.nr_gens):
                gens = [agent.gen for agent in agents if agent.gen == gen]
                if len(gens) == 0:
                    self.track_avg_population_size[gen].append(-1)
                else:
                    unique, counts = np.unique(gens, return_counts=True)
                    self.track_avg_population_size[gen].append(np.mean(counts))
        else:
            for gen in range(self.nr_gens):
                self.track_avg_population_size[gen].append(-1)

    def _update_avg_population_fitness(self, agents):
        """ Update the average fitness/reward of all entities """
        if agents:
            for gen in range(self.nr_gens):
                fitness = [agent.reward for agent in agents if agent.gen == gen]
                if len(fitness) == 0:
                    self.track_avg_population_fitness[gen].append(-1)
                else:
                    self.track_avg_population_fitness[gen].append(np.mean(fitness))
        else:
            for gen in range(self.nr_gens):
                self.track_avg_population_fitness[gen].append(-1)

    def _update_avg_population_age(self, agents):
        """ Update the average age of all entities """
        if agents:
            for gen in range(self.nr_gens):
                pop_age = [agent.age for agent in agents if agent.gen == gen]
                if len(pop_age) == 0:
                    self.track_avg_population_age[gen].append(-1)
                else:
                    self.track_avg_population_age[gen].append(np.mean(pop_age))
        else:
            for gen in range(self.nr_gens):
                self.track_avg_population_age[gen].append(-1)

    def _update_avg_best_fitness(self, agents):
        """ Update the best fitness of all entities """
        if agents:
            for gen in range(self.nr_gens):

                fitness = [agent.reward for agent in agents if agent.gen == gen]
                if len(fitness) == 0:
                    self.track_avg_best_fitness[gen].append(-1)
                else:
                    self.track_avg_best_fitness[gen].append(max(fitness))
        else:
            for gen in range(self.nr_gens):
                self.track_avg_best_fitness[gen].append(-1)

    def _update_avg_best_age(self, agents):
        """ Update the best fitness of all entities """
        if agents:
            for gen in range(self.nr_gens):
                ages = [agent.age for agent in agents if agent.gen == gen]
                if len(ages) == 0:
                    self.track_avg_best_age[gen].append(-1)
                else:
                    self.track_avg_best_age[gen].append(max([agent.age for agent in agents if agent.gen == gen]))
        else:
            for gen in range(self.nr_gens):
                self.track_avg_best_age[gen].append(-1)

    def _update_avg_best_size(self, agents):
        """ Update the best fitness of all entities """
        """ Track the average size of all populations """
        if agents:
            for gen in range(self.nr_gens):
                gens = len([agent.gen for agent in agents if agent.gen == gen])
                if gens == 0:
                    self.track_avg_best_size[gen].append(-1)
                else:
                    self.track_avg_best_size[gen].append(gens)
        else:
            for gen in range(self.nr_gens):
                self.track_avg_best_size[gen].append(-1)

    def _update_oldest_entities(self, agents):
        """ Update the oldest entity """
        if agents:
            oldest_entity = max([agent.age for agent in agents])
            self.oldest_entities.append(oldest_entity)
        else:
            self.oldest_entities.append(-1)

    def _update_best_fitness(self, agents):
        for agent in agents:
            if agent.fitness > self.best_fitness:
                self.best_fitness = agent.fitness

        if agents:
            best_fitness = max([agent.fitness for agent in agents])
            self.best_fitness_list.append(best_fitness)
        else:
            self.best_fitness_list.append(-1)

    def _get_avg(self, a_list):
        a_list = [val for val in a_list[-self.print_interval:] if val > -1]
        result = round(np.mean(a_list), 1)
        return result

    def _aggregate(self, a_list):
        aggregation = np.mean([val for val in a_list[-self.print_interval:] if val > -1])
        del a_list[:]
        return aggregation

    def update_results(self, agents, n_epi, actions, fitness):
        # Average the results between and within episodes
        self._update_avg_population_size(agents)
        self._update_avg_population_age(agents)
        self._update_avg_population_fitness(agents)
        self._update_avg_nr_attacks(agents)
        self._update_avg_nr_populations(agents)

        # Average results between episodes and take the max within episodes
        self._update_avg_best_age(agents)
        self._update_avg_best_size(agents)
        self._update_avg_best_fitness(agents)

        # Update highest
        self._update_oldest_entities(agents)
        self._update_best_fitness(agents)

        if n_epi % self.print_interval == 0 and n_epi != 0:

            for gen in range(self.nr_gens):
                self.avg_population_size[gen].append(self._aggregate(self.track_avg_population_size[gen]))
                self.avg_population_age[gen].append(self._aggregate(self.track_avg_population_age[gen]))
                self.avg_population_fitness[gen].append(self._aggregate(self.track_avg_population_fitness[gen]))
                self.avg_best_age[gen].append(self._aggregate(self.track_avg_best_age[gen]))
                self.avg_best_size[gen].append(self._aggregate(self.track_avg_best_size[gen]))
                self.avg_best_fitness[gen].append(self._aggregate(self.track_avg_best_fitness[gen]))

                self.avg_nr_attacks[gen].append(self._aggregate(self.track_avg_nr_attacks[gen]))
            self.avg_nr_populations.append(self._aggregate(self.track_avg_nr_populations))

            if self.interactive:

                if self.google_colab:
                    self._plot_google()

                else:
                    x = np.arange(self.print_interval, (len(self.avg_population_age[0]) * self.print_interval) + 1,
                                  self.print_interval)
                    for gen in range(self.nr_gens):
                        self.ax[0][0].plot(x, self.avg_population_age[gen], label=gen)
                        self.ax[0][1].plot(x, self.avg_population_fitness[gen], label=gen)
                        self.ax[0][2].plot(x, self.avg_population_size[gen], label=gen)
                        self.ax[1][0].plot(x, self.avg_best_age[gen], label=gen)
                        self.ax[1][1].plot(x, self.avg_best_fitness[gen], label=gen)
                        self.ax[1][2].plot(x, self.avg_best_size[gen], label=gen)
                        self.ax[2][0].plot(x, self.avg_nr_attacks[gen], label=gen)

                    plt.draw()
                    mypause(0.0001)

    def _plot_google(self):
        x = np.arange(self.print_interval, (len(self.avg_population_age[0]) * self.print_interval) + 1,
                      self.print_interval)

        with self.grid.output_to(0, 0):
            self.grid.clear_cell()
            plt.figure(figsize=(3, 3))
            plt.title("Average Population Age")

            for gen in range(self.nr_gens):
                plt.plot(x, self.avg_population_age[gen], label=str(gen))
            plt.legend()

        with self.grid.output_to(0, 1):
            self.grid.clear_cell()
            plt.figure(figsize=(3, 3))
            plt.title("Average Population Fitness")

            for gen in range(self.nr_gens):
                plt.plot(x, self.avg_population_fitness[gen], label=str(gen))
            plt.legend()

        with self.grid.output_to(0, 2):
            self.grid.clear_cell()
            plt.figure(figsize=(3, 3))
            plt.title("Average Population Size")

            for gen in range(self.nr_gens):
                plt.plot(x, self.avg_population_size[gen], label=str(gen))
            plt.legend()

        with self.grid.output_to(1, 0):
            self.grid.clear_cell()
            plt.figure(figsize=(3, 3))
            plt.title("Best Population Age")

            for gen in range(self.nr_gens):
                plt.plot(x, self.avg_best_age[gen], label=str(gen))
            plt.legend()

        with self.grid.output_to(1, 1):
            self.grid.clear_cell()
            plt.figure(figsize=(3, 3))
            plt.title("Best Population Fitness")

            for gen in range(self.nr_gens):
                plt.plot(x, self.avg_best_fitness[gen], label=str(gen))
            plt.legend()

        with self.grid.output_to(1, 2):
            self.grid.clear_cell()
            plt.figure(figsize=(3, 3))
            plt.title("Best Population Size")

            for gen in range(self.nr_gens):
                plt.plot(x, self.avg_best_size[gen], label=str(gen))
            plt.legend()

        with self.grid.output_to(2, 0):
            self.grid.clear_cell()
            plt.figure(figsize=(3, 3))
            plt.title("Average Nr Attacks")

            for gen in range(self.nr_gens):
                plt.plot(x, self.avg_nr_attacks[gen], label=str(gen))
            plt.legend()

        with self.grid.output_to(2, 1):
            self.grid.clear_cell()
            plt.figure(figsize=(3, 3))
            plt.title("Average Nr Populations")
            plt.plot(x, self.avg_nr_populations)


def mypause(interval):
    """ Needed to keep visualizatio minimized:

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
