import pygame

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