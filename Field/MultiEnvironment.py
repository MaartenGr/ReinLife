import numpy as np

# Gym
import gym
from gym import spaces
import pygame

# Custom
from Field.utils import check_pygame_exit
from Field import Food, Agent


class MultiEnvironment(gym.Env):
    def __init__(self, width=30, height=30, mode='computer', nr_agents=10):

        # Coordinate information
        self.width = width
        self.height = height
        self.grid_size = 16
        self.tile_location = np.random.randint(0, 2, (self.height, self.width))

        # Trackers
        self.entities = {"Food": [],
                         "Agent": []}
        self.object_types = {"Food": Food,
                             "Agent": Agent}
        self.is_render = False

        self.current_step = 0
        self.max_step = 30
        self.scores = []
        self.mode = mode
        self.nr_agents = nr_agents

        # For Gym
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=(7, 7),
                                            dtype=np.float)

    def reset(self, is_render=False):
        """ Reset the environment to the beginning """
        self.is_render = is_render
        self._reset_variables()
        self._init_objects()
        obs = [self.get_fov(agent) for agent in self.entities["Agents"]]
        return obs

    def step(self, actions):
        """ move a single step """
        self.current_step += 1
        self._act(actions)

        rewards, dones, infos = zip(*[self._get_reward(agent) for agent in self.entities["Agents"]])

        # Add food randomly
        if len([1 for food in self.entities["Food"] if food.value == 1]) <= 500:
            for i in range(3):
                if np.random.random() < 0.2:
                    self._add_entity(random_location=True, value=1, entity_type="Food")

        # Add poison randomly
        if len([1 for food in self.entities["Food"] if food.value == -1]) <= 500:
            for i in range(3):
                if np.random.random() < 0.2:
                    self._add_entity(random_location=True, value=-1, entity_type="Food")

        obs = [self.get_fov(agent) for agent in self.entities["Agents"]]

        return obs, rewards, dones, infos

    def render(self):
        """ Render the game using pygame """
        self._init_pygame_screen()
        self._step_human()
        self._draw()

        return check_pygame_exit()

    def _act(self, actions):
        """ Make the agent act and reduce its health with each step """
        for agent, action in zip(self.entities["Agents"], actions):
            if action > 3:
                self._attack(action, agent)

        for agent, action in zip(self.entities["Agents"], actions):
            if action <= 3:
                self._move(action, agent)

        for agent, action in zip(self.entities["Agents"], actions):
            self._eat(agent)

        for agent, action in zip(self.entities["Agents"], actions):
            agent.health -= 10

    def _eat(self, agent):
        closest_food = self._get_closest_food_pellet(agent)

        if agent.coordinates == closest_food.coordinates:
            if closest_food.value == 1:
                agent.health += 50
                agent.update_action("Eat Food")
                self.entities["Food"].remove(closest_food)
            elif closest_food.value == -1:
                agent.health -= 20
                agent.update_action("Eat Poison")
                self.entities["Food"].remove(closest_food)

    def _attack(self, action, agent):
        for other_agent in self.entities["Agents"]:

            if action == 4:  # attack right
                if agent.coordinates == (other_agent.x - 1, other_agent.y):
                    other_agent.health -= 100

            elif action == 5:  # attack left
                if agent.coordinates == (other_agent.x + 1, other_agent.y):
                    other_agent.health -= 100

            elif action == 6:  # attack top
                if agent.coordinates == (other_agent.x, other_agent.y + 1):
                    other_agent.health -= 100

            elif action == 7:  # attack bottom
                if agent.coordinates == (other_agent.x, other_agent.y - 1):
                    other_agent.health -= 100

    def _init_pygame_screen(self):
        """ Initialize the pygame screen for rendering """
        pygame.init()
        self.screen = pygame.display.set_mode((round(self.width) * self.grid_size, round(self.height) * self.grid_size))
        clock = pygame.time.Clock()
        clock.tick(3)
        self.screen.fill((255, 255, 255))
        self._draw_tiles()

    def get_fov(self, agent):
        """ Update the agent's field of view """
        fov_food = self._get_fov_per_entity_type(agent, entity_type="Food")
        fov_agents = self._get_fov_per_entity_type(agent, entity_type="Agents")
        fov = np.array(list(fov_food.flatten()) + list(fov_agents.flatten()) + [agent.health] + [agent.x] + [agent.y])
        return fov

    def _get_fov_matrix(self, agent):
        """ Update the agent's field of view """
        fov_food = self._get_fov_per_entity_type(agent, entity_type="Food")
        fov_agents = self._get_fov_per_entity_type(agent, entity_type="Agents")
        return fov_food, fov_agents

    def _get_fov_per_entity_type(self, agent, entity_type):
        """ Get the field of view for a single agent and a specific entity type """
        fov = np.zeros([7, 7])
        for entity in self.entities[entity_type]:

            # Make sure that only other agents are selected that are alive
            if (entity_type == "Agents" and agent != entity and not agent.dead) or entity_type != "Agents":

                # Get closest entities
                if abs(entity.x - agent.x) <= 3 and abs(entity.y - agent.y) <= 3:
                    diff_x = entity.x - agent.x
                    diff_y = agent.y - entity.y
                    fov[3 - diff_y, 3 + diff_x] = entity.value

                # Look through wall if it is on the left side
                elif agent.x <= 3 and abs(entity.y - agent.y) <= 3 and self.width - (entity.x - agent.x) <= 3:
                    diff_y = agent.y - entity.y
                    diff_x = 3 - (self.width - (entity.x - agent.x))
                    fov[3 - diff_y, diff_x] = entity.value

                # Look through wall if it is on the right side
                elif agent.x >= (self.width - 3) and abs(entity.y - agent.y) <= 3 and self.width - (agent.x - entity.x) <= 3:
                    diff_y = agent.y - entity.y
                    diff_x = 3 + (self.width - (agent.x - entity.x))
                    fov[3 - diff_y, diff_x] = entity.value

                # Look through wall if it is on the bottom (y-inverted)
                elif agent.y >= (self.height - 3) and abs(entity.x - agent.x) <= 3 and self.height - (agent.y - entity.y) <= 3:
                    diff_y = 3 + (self.height - (agent.y - entity.y))
                    diff_x = 3 + (entity.x - agent.x)
                    fov[diff_y, diff_x] = entity.value

                # Look through wall if it is on top (y-inverted)
                elif agent.y <= 3 and abs(entity.x - agent.x) <= 3 and self.height - (entity.y - agent.y) <= 3:
                    diff_y = 3 - (self.height - (entity.y - agent.y))
                    diff_x = 3 + (entity.x - agent.x)
                    fov[diff_y, diff_x] = entity.value

        return fov

    def _init_objects(self):
        """ Initialize the objects """
        # Add agent
        self.entities["Agents"] = [self._add_entity(random_location=True, value=1, entity_type="Agent")
                                   for _ in range(self.nr_agents)]

        # Add Good Food
        for i in range(self.width*self.height):
            if np.random.random() < 0.05:
                self._add_entity(random_location=True, value=1, entity_type="Food")

        # Add Bad Food
        for i in range(self.width*self.height):
            if np.random.random() < 0.05:
                self._add_entity(random_location=True, value=-1, entity_type="Food")

    def _reset_variables(self):
        """ Reset variables back their starting values """
        self.entities = {"Food": [],
                         "Agent": []}
        self.current_step = 0

    def _add_entity(self, x=None, y=None, value=0, entity_type=None, random_location=False):
        """ Add an entity to a specified (x, y) coordinate. If random_location = True, then
            the entity will be added to a random unoccupied location. """
        if random_location:
            for i in range(20):
                x = np.random.randint(self.width)
                y = np.random.randint(self.height)
                entity = self.object_types[entity_type](x, y, value)

                if not self._coordinate_is_occupied(entity.coordinates):
                    self.entities[entity_type].append(entity)
                    return entity
        else:
            entity = self.object_types[entity_type](x, y, value)
            self.entities[entity_type].append(entity)
            return entity

    def _coordinate_is_occupied(self, coordinates):
        """ Check if coordinate is occupied """
        for entity_type in self.entities.keys():
            for entity in self.entities[entity_type]:
                if coordinates == entity.coordinates:
                    return True
        return False

    def _get_closest_food_pellet(self, agent):
        """  Get the closest food pellet to the agent """
        distances = [abs(food.x-agent.x) + abs(food.y-agent.y) for food in self.entities["Food"]]
        if distances:
            idx_closest_distance = int(np.argmin(distances))
        else:
            return Food(-1, -1, 0)
        return self.entities["Food"][idx_closest_distance]

    def _move(self, action, agent):
        """ Move the agent to one space adjacent (up, right, down, left) """
        if agent.health > 0:

            # Up
            if action == 0:
                if agent.y == self.height - 1:
                    agent.update(agent.x, 0)
                else:
                    agent.update(agent.x, agent.y + 1)

            # Right
            elif action == 1:
                if agent.x == self.width - 1:
                    agent.update(0, agent.y)
                else:
                    agent.update(agent.x + 1, agent.y)

            # Down
            elif action == 2:
                if agent.y == 0:
                    agent.update(agent.x, self.height - 1)
                else:
                    agent.update(agent.x, agent.y - 1)

            # Left
            elif action == 3:
                if agent.x == 0:
                    agent.update(self.width - 1, agent.y)
                else:
                    agent.update(agent.x - 1, agent.y)

    def _get_reward(self, agent):
        """ Extract reward and whether the game has finished """
        reward = 0
        done = False
        previous_action = agent.get_action()
        info = ""

        if not agent.dead:
            if agent.health <= 0:
                agent.dead = True
                reward -= 400
                done = True
                self._add_entity(x=agent.x, y=agent.y, value=1, entity_type="Food")
                agent.update(-1, -1)
                info = "Dead"
            elif self.current_step == self.max_step:
                done = True
                reward = 400
            elif previous_action == "Eat Food":
                reward = 300
                reward = 0
            elif previous_action == "Eat Poison":
                reward -= 300
                reward = 0

        else:
            done = True

        return reward, done, info

    def _get_tiles(self):
        """ Load tile images """
        tile_1 = pygame.image.load(r'Sprites/tile_green_dark_grass.png')
        tile_2 = pygame.image.load(r'Sprites/tile_green_light_grass.png')
        return [tile_1, tile_2]

    def _draw_tiles(self):
        """ Draw tiles on screen """
        tiles = self._get_tiles()
        for i in range(self.width):
            for j in range(self.height):
                self.screen.blit(tiles[self.tile_location[j, i]], (i*16, j*16))

    def _draw(self):
        """ Draw all sprites and tiles """
        # Draw agent
        agent_img = pygame.image.load(r'Sprites/agent.png')

        for agent in self.entities["Agents"]:
            if not agent.dead:
                self.screen.blit(agent_img, (agent.x * 16, agent.y * 16))

        # Draw food
        apple = pygame.image.load(r'Sprites/apple.png')
        poison = pygame.image.load(r'Sprites/poison.png')

        for food in self.entities["Food"]:
            if food.value == 1:
                self.screen.blit(apple, (food.x * 16, food.y * 16))
            elif food.value == -1:
                self.screen.blit(poison, (food.x * 16, food.y * 16))

        pygame.display.update()

    def _step_human(self):
        """ Execute an action manually """
        if self.mode == 'human':
            events = pygame.event.get()
            action = 4
            actions = [10 for _ in range(20)]
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        actions[0] = 2
                    if event.key == pygame.K_RIGHT:
                        actions[0] = 1
                    if event.key == pygame.K_DOWN:
                        actions[0] = 0
                    if event.key == pygame.K_LEFT:
                        actions[0] = 3
                    self.step(actions)

                    fov_food, fov_agents = self._get_fov_matrix(self.entities["Agents"][0])

                    print(f"Health : {self.entities['Agents'][0].health}")
                    print(f"Dead: {str(self.entities['Agents'][0].dead)}")
                    print(f"Enemies: {fov_agents}")
