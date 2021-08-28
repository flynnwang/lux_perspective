import math, sys
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate



class LuxGame(Game):

  @property
  def opponent_id(self):
    return (self.id + 1) % 2

  @property
  def player(self):
    return self.players[self.id]

  @property
  def opponent(self):
    return self.players[self.opponent_id]

  def update(self, observation, configuration):
    game_state = self

    ### Do not edit ###
    if observation["step"] == 0:
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])


class Strategy:

  def __init__(self):
    self.actions = []
    self.game = LuxGame()

  def update(self, observation, configuration):
    self.game.update(observation, configuration)

    # Clear up actions for current step.
    self.actions = []

  def execute(self):
    g = self.game
    actions = self.actions

    ### AI Code goes down here! ###
    player = g.player
    opponent = g.players[self.game.opponent_id]
    width, height = g.map.width, g.map.height

    resource_tiles: list[Cell] = []
    for y in range(height):
        for x in range(width):
            cell = g.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)

    for _, city in player.cities.items():
      for citytile in city.citytiles:
        if citytile.can_act():
          actions.append(citytile.build_worker())

    # we iterate over all our units and do something with them
    for unit in player.units:
      if unit.is_worker() and unit.can_act():
        closest_dist = math.inf
        closest_resource_tile = None
        if unit.get_cargo_space_left() > 0:
            # if the unit is a worker and we have space in cargo, lets find the nearest resource tile and try to mine it
            for resource_tile in resource_tiles:
              if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
              if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
              dist = resource_tile.pos.distance_to(unit.pos)
              if dist < closest_dist:
                  closest_dist = dist
                  closest_resource_tile = resource_tile
            if closest_resource_tile is not None:
                actions.append(unit.move(unit.pos.direction_to(closest_resource_tile.pos)))
        else:
          if (g.turn // 20) % 2 == 0 and unit.can_build(g.map):
            actions.append(unit.build_city())
            continue

          # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
          if len(player.cities) > 0:
            closest_dist = math.inf
            closest_city_tile = None
            for k, city in player.cities.items():
              for city_tile in city.citytiles:
                dist = city_tile.pos.distance_to(unit.pos)
                if dist < closest_dist:
                  closest_dist = dist
                  closest_city_tile = city_tile
            if closest_city_tile is not None:
              move_dir = unit.pos.direction_to(closest_city_tile.pos)
              actions.append(unit.move(move_dir))




_strategy = Strategy()

def agent(observation, configuration):
  _strategy.update(observation, configuration)
  _strategy.execute()
  return _strategy.actions
