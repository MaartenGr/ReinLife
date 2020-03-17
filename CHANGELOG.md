# Changelog

All notable changes to this project will be documented in this file. This changelog
follows the following style:
- Added
- Changed
- Fixed
- Removed


## [0.2.2] - 2020-03-13

### Fixed
- The fov for entities and their health was incorrect and now fixed.  

## [0.2.1] - 2020-03-13

### Added
- Added changelog 

### Changed
- Major overhaul of the readme

### Fixed
- Improved reward function
    - From:
        - 5 + (5 * sum([1 for other_agent in self.agents if agent.gen == other_agent.gen]))

    - To
        - sum([other_agent.health / max_age for other_agent in self.agents if agent.gen == other_agent.gen])

## [0.2.0] - 2020-03-13

### Added
- Implemented A2C, DQN, PER-DQN, and PPO (although it does has some stability issues)
- Added genes, additional observation space, asexual reproduction

### Changed
- Major restructure of packages

## [0.1.0] - 2020-03-02

### Added
* Created a Sprite Generator which is currently not used
* Implemented infiniteworld with fov through borders
* Agents can attack or move (8 actions, attack in direction or move in direction)
* PPO (+LSTM) and DQN work

### Changed
* Did a rework of the environment which led to a speed-up of 7x


