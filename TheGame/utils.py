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
            pygame.draw.line(self.surface, BLACK, (GRID_COORD_MARGIN_SIZE, liCoord), (self.surface.get_width(), liCoord))
        for co in range(self.colNb + 1):
            colCoord = GRID_COORD_MARGIN_SIZE + co * CELL_SIZE
            pygame.draw.line(self.surface, BLACK, (colCoord, GRID_COORD_MARGIN_SIZE), (colCoord,self.surface.get_height()))


def check_pygame_exit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return False
    return True


class Results:
    def __init__(self, print_interval, interactive=False, save_visualization=False):
        self.avg_population_size = []
        self.avg_population_age = []

        self.track_avg_population_size = []
        self.track_avg_population_age = []

        self.track_nr_actions = []

        self.oldest_entities = []
        self.max_age_list = []
        self.best_scores_list = []
        self.best_score = -1_000_000
        self.print_interval = print_interval
        self.interactive = interactive
        self.save_visualization = save_visualization

        if interactive:
            # Plotting
            plt.ion()
            self.fig, (self.ax1, self.ax2) = plt.subplots(2)
            plt.show(block=False)

    def _update_population_size(self, agents):
        if agents:
            gens = [agent.gen for agent in agents]
            unique, counts = np.unique(gens, return_counts=True)
            self.track_avg_population_size.append(np.mean(counts))
        else:
            self.track_avg_population_size.append(-1)

    def _update_population_age(self, agents):
        if agents:
            avg_pop_size = np.mean([agent.age for agent in agents])
            self.track_avg_population_age.append(avg_pop_size)
        else:
            self.track_avg_population_age.append(-1)

    def _update_oldest_entities(self, agents):
        if agents:
            oldest_entity = max([agent.age for agent in agents])
            self.oldest_entities.append(oldest_entity)
        else:
            self.oldest_entities.append(-1)

    def _update_best_score(self, agents):
        for agent in agents:
            if agent.fitness > self.best_score:
                self.best_score = agent.fitness

        if agents:
            best_score = max([agent.fitness for agent in agents])
            self.best_scores_list.append(best_score)
        else:
            self.best_scores_list.append(-1)

    def _get_avg(self, a_list):
        a_list = [val for val in a_list[-self.print_interval:] if val > -1]
        result = round(np.mean(a_list), 1)
        return result

    def _add_avg_last_n(self, a_list):
        a_list = [val for val in a_list[-self.print_interval:] if val > -1]
        result = round(np.mean(a_list), 1)
        return result

    def _aggregate(self, a_list):
        aggregation = np.mean([val for val in a_list[-self.print_interval:] if val > -1])
        del a_list[:]
        return aggregation

    def update_results(self, agents, n_epi, actions):
        self._update_population_size(agents)
        self._update_population_age(agents)
        self._update_oldest_entities(agents)
        self._update_best_score(agents)
        if len(actions) != 0:
            self.track_nr_actions.append(len([action for action in actions if action >= 4])/len(actions))
        else:
            self.track_nr_actions.append(0)

        if n_epi % self.print_interval == 0 and n_epi != 0:
            self.avg_population_age.append(self._aggregate(self.track_avg_population_age))
            self.avg_population_size.append(self._aggregate(self.track_avg_population_size))

            print(f"######## Results ######## \n"
                  f"Episode: {n_epi}  \n"
                  f"Avg pop size: {self.avg_population_age[-1]}  \n"
                  f"Avg age: {self.avg_population_age[-1]}  \n"
                  f"Avg oldest: {self._get_avg(self.oldest_entities)}  \n"
                  f"Avg best scores: {self._get_avg(self.best_scores_list)} \n"
                  f"Best score: {self.best_score}  \n"
                  f"Avg nr attacks: {np.mean(self.track_nr_actions[-self.print_interval:])}  \n")

            if self.interactive:
                self.ax1.plot(range(len(self.avg_population_age)), self.avg_population_age, 'r-')
                self.ax2.plot(range(len(self.avg_population_size)), self.avg_population_size, 'r-')
                plt.draw()
                # plt.pause(0.0001)
                mypause(0.0001)
                # self.fig.canvas.start_event_loop(0.001)

            # if self.save_visualization:
            #     fig, (ax1, ax2) = plt.subplots(2)
            #     self.ax1.plot(range(len(self.avg_population_age)), self.avg_population_age, 'r-')
            #     self.ax2.plot(range(len(self.avg_population_size)), self.avg_population_size, 'r-')


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
