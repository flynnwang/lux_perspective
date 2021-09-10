"""

* add cell near city as target.

Total Matches: 429 | Matches Queued: 7
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
/Users/flynnwang/dev/playground/lux_perspective/main.py | vdOYoX49CNJq   | 25.1753266      | μ=27.615, σ=0.813  | 234
/Users/flynnwang/dev/playground/versions/explore_cluster/main.py | M3bmkebwG4Cx   | 25.0145022      | μ=27.452, σ=0.812  | 218
/Users/flynnwang/dev/playground/versions/boost_near_resource/main.py | V0q7LGOpDy0b   | 20.9430865      | μ=23.375, σ=0.810  | 220
/Users/flynnwang/dev/playground/versions/Tong_Hui_Kang/main.py | iKNww9lYVkHe   | 19.1461051      | μ=21.682, σ=0.845  | 186
opponent_name	Tong_Hui_Kang	boost_near_resource	explore_cluster	lux_perspective
Tong_Hui_Kang	0.000	30.769	18.644	20.968
boost_near_resource	69.231	0.000	27.143	24.390
explore_cluster	81.356	72.857	0.000	44.828
lux_perspective	79.032	75.610	55.172	0.000


* check transfer condition and do transfer.
* WIP do not goto the cell of a transfer action worker
* Do not send money to citytile, but build

Total Matches: 198 | Matches Queued: 10
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
/Users/flynnwang/dev/playground/lux_perspective/main.py | rKWMEoftKxLP   | 25.4902518      | μ=28.230, σ=0.913  | 105
/Users/flynnwang/dev/playground/versions/explore_cluster/main.py | 6tlVPTUCce9a   | 23.9821881      | μ=26.720, σ=0.913  | 103
/Users/flynnwang/dev/playground/versions/boost_near_resource/main.py | nmsDh2ZwEu1c   | 22.4263394      | μ=25.188, σ=0.921  | 98
/Users/flynnwang/dev/playground/versions/Tong_Hui_Kang/main.py | F3NuuZ1BN8Ja   | 18.1242076      | μ=21.228, σ=1.035  | 90

opponent_name	Tong_Hui_Kang	boost_near_resource	explore_cluster	lux_perspective
Tong_Hui_Kang	0.000	16.000	21.875	18.750
boost_near_resource	88.000	0.000	38.889	30.556
explore_cluster	78.125	61.111	0.000	42.857
lux_perspective	81.250	69.444	57.143	0.000


WIP
* save more citytiles, how to move at night: need to consider city assingment.

* Do not tranfer at night
* Assign maximum amount of resource worker per cluster.




* debug 437892541 at turn 49, worker going back.


* do not move onto build action unit.
* give multiple worker to coal & uranium cell
* worker forward fuel (from coal and uranium) to city better


# TODO
* give some weight to not finished research resource (arrive at them earlier)
* Add resource weight to city tile.
* Raise priority of near_resource_tile targeted worker

TODO:
* Add units (enemy and mine) to cell and check blocking
# TODO: build city away from near resource tile


- Learning

* unit cargo won't goto 100 at night
"""

import math, sys
from collections import defaultdict, deque

import numpy as np
import scipy.optimize

from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
from utility import *

DEBUG = True


# TODO: add more
BUILD_CITYTILE_ROUND = 26

MAX_PATH_WEIGHT = 99999


MAX_UNIT_NUM = 60


def dist_decay(dist, game_map):
  # decay = game_map.width
  decay =  5
  # decay = 24
  return (dist / decay) + 1


def worker_enough_cargo_to_build(worker, collected_amount):
  return (worker_total_cargo(worker) + collected_amount >= CITY_BUILD_COST)


def get_neighbour_positions(pos, game_map, return_cell=False):
  check_dirs = [
    DIRECTIONS.NORTH,
    DIRECTIONS.EAST,
    DIRECTIONS.SOUTH,
    DIRECTIONS.WEST,
  ]
  positions = []
  for direction in check_dirs:
    newpos = pos.translate(direction, 1)
    if not is_within_map_range(newpos, game_map):
      continue

    if return_cell:
      newpos = game_map.get_cell_by_pos(newpos)
    positions.append(newpos)
  return positions

N9_DIRS = [(-1, 1), (0, 1), (1, 1),
           (-1, 0), (1, 0),
           (-1, -1), (0, -1), (1, -1)]

def get_nb9_positions(pos, game_map):
  positions = []
  for dx, dy in N9_DIRS:
    newpos = Position(pos.x+dx, pos.y+dy)
    if is_within_map_range(newpos, game_map):
      positions.append(newpos)
  return positions


def cell_has_opponent_citytile(cell, game):
  citytile = cell.citytile
  return citytile is not None and citytile.team == game.opponent.team

def cell_has_player_citytile(cell, game):
  citytile = cell.citytile
  return citytile is not None and citytile.team == game.player.team


def is_resource_researched(resource, player):
  if (resource.type == Constants.RESOURCE_TYPES.COAL
      and not player.researched_coal()):
    return False
  if (resource.type == Constants.RESOURCE_TYPES.URANIUM
      and not player.researched_uranium()):
    return False
  return True


def get_cell_resource_values(cell, player, unit=None):
  # (amount, fuel)
  if not cell.has_resource():
    return 0, 0

  resource = cell.resource
  if not is_resource_researched(resource, player):
    return 0, 0

  amount = get_worker_collection_rate(resource)
  amount = min([amount, resource.amount])
  if unit:
    amount = min(amount, unit.get_cargo_space_left())
  fuel = amount * get_resource_to_fuel_rate(resource)
  return amount, fuel

# TODO(wangfei): try more accurate estimate
def estimate_resource_night_count(resource, upkeep):
  cargo = resource_to_cargo(resource)
  return cargo_night_endurance(cargo, upkeep)


def estimate_cell_night_count(cell, upkeep, game_map):
  nights = estimate_resource_night_count(cell.resource, upkeep)
  for newpos in get_neighbour_positions(cell.pos, game_map):
    nb_cell = game_map.get_cell_by_pos(newpos)
    nights += estimate_resource_night_count(nb_cell.resource, upkeep)
  return nights


def get_one_step_collection_values(cell, player, game_map):
  amount, fuel = get_cell_resource_values(cell, player)
  for newpos in get_neighbour_positions(cell.pos, game_map):
    nb_cell = game_map.get_cell_by_pos(newpos)
    a, f = get_cell_resource_values(nb_cell, player)
    amount += a
    fuel += f
  return amount, fuel


class LuxGame(Game):

  @property
  def game_map(self):
    return self.map

  @property
  def opponent_id(self):
    return (self.id + 1) % 2

  @property
  def player(self):
    return self.players[self.id]

  @property
  def opponent(self):
    return self.players[self.opponent_id]

  @property
  def player_city_tiles(self):
    return [citytile
            for _, city in self.player.cities.items()
            for citytile in city.citytiles]

  @property
  def is_night(self):
    return not is_day(self.turn)

  @property
  def is_day(self):
    return is_day(self.turn)


  def update(self, observation, configuration):
    game_state = self

    ### Do not edit ###
    if observation["step"] == 0:
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])


# TODO(wangei): use priority based search?
class ShortestPath:

  def __init__(self, game, start_pos, forbidden_positions):
    self.game = game
    self.game_map = game.game_map
    self.start_pos = start_pos
    self.forbidden_positions = forbidden_positions
    self.dist = np.ones((self.game_map.width, self.game_map.height)) * MAX_PATH_WEIGHT

  def compute(self):
    q = deque([self.start_pos])
    self.dist[self.start_pos.x, self.start_pos.y] = 0

    while q:
      cur = q.popleft()
      cur_dist = self.dist[cur.x, cur.y]
      for newpos in get_neighbour_positions(cur, self.game_map):
        nb_cell = self.game_map.get_cell_by_pos(newpos)

        if cell_has_opponent_citytile(nb_cell, self.game):
          continue

        # Skip player unit in cooldown.
        if (nb_cell.unit and
            nb_cell.unit.team == self.game.opponent_id and not nb_cell.unit.can_act()):
          continue
        if newpos in self.forbidden_positions:
          continue

        nb_dist = self.dist[newpos.x, newpos.y]
        if cur_dist + 1 >= nb_dist:
          continue

        self.dist[newpos.x, newpos.y] = cur_dist + 1
        q.append(newpos)
        # print(f' start_from {self.start_pos}, append {newpos}, cur_dist={cur_dist}', file=sys.stderr)


  def shortest_dist(self, pos):
    return self.dist[pos.x, pos.y]

  def path_blocked_by_citytile(self, target_pos):
    path_positions = {}

    target_dist = self.shortest_dist(target_pos)
    if target_dist >= MAX_PATH_WEIGHT:
      return True

    if self.game_map.get_cell_by_pos(target_pos).citytile is not None:
      return True

    q = deque([target_pos])
    path_positions[target_pos] = self.dist[target_pos.x, target_pos.y]
    while q:
      cur = q.popleft()
      cur_dist = self.dist[cur.x, cur.y]
      for newpos in get_neighbour_positions(cur, self.game_map):
        nb_dist = self.dist[newpos.x, newpos.y]
        if nb_dist == cur_dist - 1:
          if newpos in path_positions:
            continue
          if self.game_map.get_cell_by_pos(newpos).citytile is not None:
            continue

          q.append(newpos)
          path_positions[newpos] = cur_dist - 1
    return self.start_pos not in path_positions

  def compute_shortest_path_points(self, target_pos):
    path_positions = {}

    target_dist = self.shortest_dist(target_pos)
    if target_dist >= MAX_PATH_WEIGHT:
      return path_positions

    q = deque([target_pos])
    path_positions[target_pos] = self.dist[target_pos.x, target_pos.y]

    total_append = 0
    while q:
      cur = q.popleft()
      cur_dist = self.dist[cur.x, cur.y]
      for newpos in get_neighbour_positions(cur, self.game_map):
        nb_dist = self.dist[newpos.x, newpos.y]
        if nb_dist == cur_dist - 1 and newpos not in path_positions:
          total_append += 1
          q.append(newpos)
          path_positions[newpos] = cur_dist - 1

    # print(f'path points {self.start_pos}, totol_append={total_append}', file=sys.stderr)
    return path_positions



# TODO: add near resource tile to cluster
class ClusterInfo:

  def __init__(self, game):
    self.game = game
    self.game_map = game.game_map
    self.position_to_cid = np.ones((self.game_map.width, self.game_map.height)) * -1
    self.cluster_fuel = {}  # (cluster_id, cluster fuel)
    self.cluster_citytile_count = {}
    self.max_cluster_fuel = 0
    self.max_cluster_id = 0

  def set_cid(self, pos, cid):
    self.position_to_cid[pos.x][pos.y] = cid

  def get_cid(self, pos):
    return self.position_to_cid[pos.x][pos.y]

  def compute(self):
    def is_valid_resource_cell(cell):
      return (cell.has_resource()
              and is_resource_researched(cell.resource, self.game.player))

    def search_cluster(start_cell):
      vistied_citytile_positions = set()
      total_fuel = resource_fuel(start_cell.resource)
      q = deque([start_cell.pos])
      self.set_cid(start_cell.pos, max_cluster_id)

      while q:
        cur = q.popleft()
        for newpos in get_neighbour_positions(cur, self.game_map):
          nb_cell = self.game_map.get_cell_by_pos(newpos)

          # Count citytile.
          # if (nb_cell.citytile != None
              # and nb_cell.citytile.team == self.game.player.team):
          if (nb_cell.citytile != None):
            vistied_citytile_positions.add(nb_cell.citytile.pos)

          if not is_valid_resource_cell(nb_cell):
            continue

          cid = self.get_cid(newpos)
          if cid >= 0:
            continue

          self.set_cid(newpos, max_cluster_id)
          total_fuel += resource_fuel(nb_cell.resource)
          q.append(nb_cell.pos)

      return total_fuel, len(vistied_citytile_positions)

    max_cluster_id = 0

    for x in range(self.game_map.width):
      for y in range(self.game_map.height):
        pos = Position(x, y)
        if self.get_cid(pos) >= 0:
          continue

        cell = self.game_map.get_cell_by_pos(pos)
        if not is_valid_resource_cell(cell):
          continue

        fuel, n_citytile = search_cluster(cell)
        self.cluster_fuel[max_cluster_id] = fuel
        self.cluster_citytile_count[max_cluster_id] = n_citytile
        max_cluster_id += 1
        self.max_cluster_fuel = max(self.max_cluster_fuel, fuel)
        # print(f'p[{pos}], fuel={fuel}', file=sys.stderr)
    self.max_cluster_id = max_cluster_id

    # print(f't={self.game.turn}, total_cluster={max_cluster_id}', file=sys.stderr)


  def query_cluster_fuel_factor(self, pos):
    cid = self.position_to_cid[pos.x][pos.y]
    if cid < 0:
      return 0
    return self.cluster_fuel[cid] / self.max_cluster_fuel


class Strategy:

  def __init__(self):
    self.actions = []
    self.game = LuxGame()

  @property
  def circle_turn(self):
    return self.game.turn % CIRCLE_LENGH

  @property
  def game_map(self):
    return self.game.game_map

  def update(self, observation, configuration):
    self.game.update(observation, configuration)

    # Clear up actions for current step.
    self.actions = []

    for unit in self.game.player.units:
      unit.has_planned_action = False
      unit.target_pos = None
      unit.transfer_build_locations = set()

      unit.unit_night_count = cargo_night_endurance(unit.cargo, get_unit_upkeep(unit))

      round_night_count = get_night_count_this_round(self.game.turn)
      unit.is_dying = unit.unit_night_count < round_night_count
      unit.cell = self.game_map.get_cell_by_pos(unit.pos)

      unit.target_cid = -1
      unit.is_transfer_worker = False

  def add_unit_action(self, unit, action):
    assert unit.has_planned_action == False

    unit.has_planned_action = True
    self.actions.append(action)


  def cell_near_done_research_resource(self, cell):
    for pos in get_neighbour_positions(cell.pos, self.game_map):
      nb_cell = self.game_map.get_cell_by_pos(pos)
      if (nb_cell.has_resource()
          and is_resource_researched(nb_cell.resource, self.game.player)):
        return True
    return False


  def update_game_map_unit_info(self):
    for y in range(self.game.map_height):
      for x in range(self.game.map_width):
        cell = self.game_map.get_cell(x, y)
        cell.is_near_resource = False
        cell.is_citytile_neighbour = False
        cell.unit = None

    for unit in self.game.player.units:
      cell = self.game_map.get_cell_by_pos(unit.pos)
      cell.unit = unit

    for unit in self.game.opponent.units:
      cell = self.game_map.get_cell_by_pos(unit.pos)
      cell.unit = unit

  def has_citytile_neighbours(self, cell):
    for newpos in get_neighbour_positions(cell.pos, self.game_map):
      nb_cell = self.game_map.get_cell_by_pos(newpos)
      if cell_has_opponent_citytile(cell, self.game):
        return True
    return False

  def assign_worker_target(self):
    MAX_UNIT_PER_CITY = 4
    g = self.game
    player = g.player

    workers = [unit for unit in player.units
               if (unit.is_worker()
                   and unit.target_pos is None
                   and not unit.has_planned_action)]

    def is_near_resource_cell(cell):
      return (not cell.has_resource()
              and cell.citytile is None
              and self.cell_near_done_research_resource(cell))

    target_cells = []
    for y in range(g.map_height):
      for x in range(g.map_width):
        cell = g.game_map.get_cell(x, y)

        is_target_cell = False
        if cell_has_player_citytile(cell, self.game):
          cell.citytile.cell = cell
          target_cells.extend([cell] * MAX_UNIT_PER_CITY)
        elif cell.has_resource():
          is_target_cell = True
        elif is_near_resource_cell(cell):
          cell.is_near_resource = True
          is_target_cell = True
        elif self.has_citytile_neighbours(cell):
          cell.is_citytile_neighbour = True
          is_target_cell = True
        elif (cell.unit and cell.unit.team == player.team):
          cell.is_worker_cell = True
          is_target_cell = True

        if is_target_cell:
          target_cells.append(cell)

    def is_resource_tile_can_save_dying_worker(resource_tile, worker):
      if (not resource_tile.has_resource()
          or (not is_resource_researched(resource_tile.resource, player))):
        return False
      if self.circle_turn >= BUILD_CITYTILE_ROUND and worker.is_dying:
        cell_nights = estimate_cell_night_count(cell, get_unit_upkeep(worker), g.game_map)
        round_nights = get_night_count_this_round(g.turn)
        if worker.unit_night_count + cell_nights >= round_nights:
          return True
      return False

    MAX_WEIGHT_VALUE = 10000
    CLUSTER_BOOST_WEIGHT = 200
    UNIT_SAVED_BY_RES_WEIGHT = 10
    def get_resource_weight(worker, resource_tile, dist, unit_night_count):
      # Use dist - 1, because then the worker will get resource.
      target_night_count = get_night_count_by_dist(g.turn, dist-1, worker.cooldown,
                                                   get_unit_action_cooldown(worker))
      # TODO: add some buffer for safe arrival
      if unit_night_count < target_night_count:
        return -1

      # Give a small weight for any resource 0.1 TODO: any other option?
      wt = 0.1

      # Try to hide next to resource grid.
      if is_resource_tile_can_save_dying_worker(resource_tile, worker):
        wt += UNIT_SAVED_BY_RES_WEIGHT

      _, fuel = get_one_step_collection_values(resource_tile, player, g.game_map)
      wt += fuel

      cid = self.cluster_info.get_cid(resource_tile.pos)
      boost_cluster = 0
      if worker.target_cid > 0 and worker.target_cid == cid:
        cluster_fuel_factor = self.cluster_info.query_cluster_fuel_factor(resource_tile.pos)
        boost_cluster += CLUSTER_BOOST_WEIGHT * cluster_fuel_factor

      # return wt / dist_decay(dist, g.game_map) + CLUSTER_BOOST_WEIGHT * cluster_fuel_factor
      return wt / (dist + 1) + boost_cluster
      # return wt / dist_decay(dist, g.game_map)

    def get_city_tile_weight(worker, city_cell, dist, unit_night_count):
      """
      1. collect fuel
      2. protect dying worker [at night]
      3. receive fuel from a fuel full worker on cell
      """
      CITYTILE_LOST_WEIGHT = 200

      # TODO: It's asuming dist are full of danger, but it could be move inside citytile.
      target_night_count = get_night_count_by_dist(g.turn, dist, worker.cooldown,
                                                   get_unit_action_cooldown(worker))
      if unit_night_count < target_night_count:
        return -1

      # amount, fuel = get_one_step_collection_values(citytile.cell, player, g.game_map)
      # wt = fuel # base score.
      citytile = city_cell.citytile
      wt = 0

      # Try to hide in the city if worker will run out of fuel at night
      city = g.player.cities[citytile.cityid]
      city_will_last = not city_wont_last_at_nights(g.turn, city)
      _, worker_collect_fuel = get_cell_resource_values(worker.cell, player, worker)
      worker_cell_fuel_enough = (worker_collect_fuel >= get_unit_upkeep(worker))

      if (self.circle_turn >= BUILD_CITYTILE_ROUND and worker.is_dying and city_will_last
          and not worker_cell_fuel_enough):
        wt += MAX_WEIGHT_VALUE / 10


      # if worker.id == 'u_12':
        # print(f"w[{worker.id}], t[{citytile.pos}], turn_c={self.circle_turn >= BUILD_CITYTILE_ROUND}, dying={worker.is_dying}, city_last={city_will_last}", file=sys.stderr)

      # TODO(wangfei): estimate worker will full

      n_citytile = len(city.citytiles)

      # Boost when worker is almost full and sit next to a city.
      days_left = DAY_LENGTH - self.circle_turn - worker.cooldown
      round_nights = get_night_count_this_round(g.turn)
      # TODO: add city won't last
      if (self.circle_turn >= BUILD_CITYTILE_ROUND
          and worker_cell_fuel_enough
          and worker.get_cargo_space_left() < 10
          and worker.pos.distance_to(city_cell.pos) == 1):
        # print(f"w[{worker.id}], t[{citytile.pos}], city_last={city_will_last}, boost city lost",
              # file=sys.stderr)
        wt += CITYTILE_LOST_WEIGHT * n_citytile

        # More weights on dying city
        # if city_left_days < round_nights:
          # wt += n_citytile * CITYTILE_LOST_WEIGHT

      # When there is not fuel on the map, go back to city
      if self.cluster_info.max_cluster_fuel == 0:
        wt += 100


      # TODO: support night
      # Save more citytile if worker has enough resource to save it
      # - enough time to arrive
      # - with substential improvement of its living
      # TODO(wangfei): this is not right, more tile will have larger light_upkeep
      unit_time_cost = (dist - 1) * get_unit_action_cooldown(worker) + 1
      city_left_days = math.ceil(city.fuel / city.light_upkeep)
      woker_fuel = worker_total_fuel(worker)
      city_left_days_deposited = math.ceil((city.fuel + woker_fuel) / city.light_upkeep)
      if (days_left >= unit_time_cost
          and city_left_days < NIGHT_LENGTH
          and city_left_days_deposited > NIGHT_LENGTH
          and city_left_days_deposited - city_left_days >= (18 // n_citytile)):
        wt += CITYTILE_LOST_WEIGHT * len(city.citytiles)

      # return wt / dist_decay(dist, g.game_map)
      return wt / (dist + 1)

      # TODO(wangfei): merge near resource tile and resource tile weight functon
    def get_near_resource_tile_weight(worker, near_resource_tile, dist,
                                      unit_night_count):
      target_night_count = get_night_count_by_dist(g.turn, dist, worker.cooldown,
                                                   get_unit_action_cooldown(worker))
      # TODO: add some buffer for safe arri0
      if unit_night_count < target_night_count:
        return -1

      amount, fuel = get_one_step_collection_values(near_resource_tile, player, g.game_map)

      wt = 0.1 + fuel

      build_city_bonus = False
      days_left = BUILD_CITYTILE_ROUND - self.circle_turn - worker.cooldown
      if (worker_enough_cargo_to_build(worker, amount)
          and self.circle_turn < BUILD_CITYTILE_ROUND
          and days_left >= (dist - 1) * get_unit_action_cooldown(worker) + 1):
        build_city_bonus = True

      # TODO: maybe time consuming
      if (dist <= 4 and self.shortet_paths[worker.id].path_blocked_by_citytile(near_resource_tile.pos)):
        build_city_bonus = False

      # blocked_by_citytile = shortest_path.
      # if not blocked_by_citytile:

      # cargo_full_rate = worker_cargo_full_rate(worker)
      # boost = (np.e ** cargo_full_rate)
      # wt *= boost

      # To build on transfer build location.
      if (near_resource_tile.pos in worker.transfer_build_locations):
        build_city_bonus = True

      # Too large the build city bonus will cause worker divergence from its coal mining position
      if build_city_bonus:
        wt += 1000

        # p = near_resource_tile.pos
        # if worker.id == 'u_1' and p in [Position(7, 3), Position(7, 1)]:
          # print(f"w[{worker.id}] - {near_resource_tile.pos}, blocked_by_citytile={blocked_by_citytile}, boost={boost}, fuel={fuel}", file=sys.stderr)

      # Try to hide next to resource grid.
      if is_resource_tile_can_save_dying_worker(near_resource_tile, worker):
        wt += UNIT_SAVED_BY_RES_WEIGHT

      # return wt / dist_decay(dist, g.map)
      # if worker.id == 'u_1' and target.pos in [Position(5, 10), Position(4, 7)]:
        # print(f"w[{worker.id}], fuel={fuel}, wt={wt}, bonus={build_city_bonus}", file=sys.stderr)
      return wt / (dist + 1)

    def get_city_tile_neighbour_weight(worker, cell):
      wt = -999
      if cell.pos in worker.transfer_build_locations:
        nb_citytiles = [nb_cell.citytile
                        for nb_cell in get_neighbour_positions(worker.pos, self.game_map, return_cell=True)
                        if (nb_cell.citytile != None
                            and nb_cell.citytile.team == self.game.player.team)]
        assert nb_citytiles > 0
        wt = 300 * nb_citytiles
      return wt

    def get_worker_tile_weight(worker, target):
      if worker.pos == target.pos:
        return 0.1
      return -1


    # Value matrix for ship target assginment
    # * row: workers
    # * column: point of interests
    weights = np.zeros((len(workers), len(target_cells)))

    # MAIN_PRINT
    print((f'*turn={g.turn}, #W={len(workers)}, #C={player.city_tile_count} '
           f'R={g.player.research_points}'),
          file=sys.stderr)
    for i, worker in enumerate(workers):
      round_night_count = get_night_count_this_round(g.turn)
      unit_night_count = worker.unit_night_count

      # if worker.id == 'u_29':
        # print(f"w[{worker.id}], unit_night_count={unit_night_count}, round_night_count={round_night_count}", file=sys.stderr)

      for j, target in enumerate(target_cells):
        # Can't arrive at target cell.
        shortest_path = self.shortet_paths[worker.id]
        dist = shortest_path.shortest_dist(target.pos)
        if dist >= MAX_PATH_WEIGHT:
          continue

        v = 0
        if cell_has_player_citytile(target, self.game):
          v = get_city_tile_weight(worker, target, dist, unit_night_count)
        elif target.has_resource():
          v = get_resource_weight(worker, target, dist, unit_night_count)
        elif target.is_near_resource:
          v = get_near_resource_tile_weight(worker, target, dist,
                                            unit_night_count)
        elif target.is_citytile_neighbour:
          v = get_city_tile_neighbour_weight(worker, target)
        else:
          v = get_worker_tile_weight(worker, target)
        weights[i, j] = v

        # if worker.id == 'u_1' and target.pos in [Position(4, 7), Position(4, 8),
                                                 # Position(4, 9)]:
          # print(f"w[{worker.id}], cd={worker.cooldown}, t[{target.pos}], wt={v:.1f}", file=sys.stderr)


    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    for worker_idx, target_idx in zip(rows, cols):
      worker = workers[worker_idx]
      target = target_cells[target_idx]
      worker.target_pos = target.pos

      # TODO: skip negative weight.
      if DEBUG:
        # print(f'worker[{worker_idx}], v={weights[worker_idx, target_idx]}', file=sys.stderr)
        # x = annotate.x(worker.pos.x, worker.pos.y)
        c = annotate.circle(target.pos.x, target.pos.y)
        if worker.target_cid > 0:
          c = annotate.x(target.pos.x, target.pos.y)

        line = annotate.line(worker.pos.x, worker.pos.y, target.pos.x, target.pos.y)
        self.actions.extend([c, line])



  def compute_worker_moves(self):
    g = self.game
    player = g.player
    workers = [unit for unit in player.units
               if unit.is_worker() and not unit.has_planned_action]

    # unit_target_positions = set()
    # for w in workers:
      # if w.target_pos:
        # unit_target_positions.add(w.target_pos)

    MAX_MOVE_WEIGHT = 9999
    def get_step_weight(worker, next_position, shortest_path,
                       shortest_path_points):
      """Only workers next 5 positions will be computed here."""
      if worker.pos == next_position:
        # If a worker can't move, it's a stone.
        if not worker.can_act():
          return MAX_MOVE_WEIGHT

        # If woker is building city.
        if worker.is_building_city:
          return MAX_MOVE_WEIGHT

      assert worker.target_pos is not None
      assert shortest_path_points is not None

      next_cell = g.game_map.get_cell_by_pos(next_position)
      # Can't move on opponent citytile.
      if (next_cell.citytile is not None
          and next_cell.citytile.team == self.game.opponent.team):
        return -MAX_MOVE_WEIGHT

      # Do not move onto a transfer worker
      if (next_cell.unit
          and next_cell.unit.team == player.team
          and next_cell.unit.is_transfer_worker):
        return -MAX_MOVE_WEIGHT


      # try not step on citytile: not good
      # if (next_cell.citytile is not None
          # and next_cell.citytile.team == g.player.team
          # and is_night(g.turn)):
        # citytile = next_cell.citytile
        # city = g.player.cities[citytile.cityid]
        # if city_wont_last_at_nights(g.turn, city):
          # v -= 0.1
      fuel = 0
      v = 0
      if next_position in shortest_path_points:
        v += 1

        target_dist = shortest_path.shortest_dist(worker.target_pos)
        worker_to_next_pos_dist = shortest_path_points[next_position]
        next_pos_to_target_dist = target_dist - worker_to_next_pos_dist
        if next_pos_to_target_dist < target_dist:
          v += 50

          # Priority all positions of th dying worker, let others make room for him.
          if worker.is_dying:
            v += 100

          cell = self.game_map.get_cell_by_pos(worker.target_pos)
          if cell.has_resource():
            v += 1

          # Try step on resource: the worker version is better, maybe because
          # other worker can use that.
          amount, fuel = get_cell_resource_values(next_cell, g.player)
          if fuel > 0:
            v += 1

          if next_position in worker.transfer_build_locations:
            v += 100

        # demote target cell one next move

      # if worker.id in ['u_3', 'u_1']:
      # print(f"w[{worker.id}]@{worker.pos}, next[{next_position}], v={v}, target={worker.target_pos}, fuel={fuel}", file=sys.stderr)

      return v

    def gen_next_positions(worker):
      if not worker.can_act():
        return [worker.pos]

      # TODO: skip non-reachable positions?
      return [worker.pos] + get_neighbour_positions(worker.pos, g.game_map)

    next_positions = {
      pos
      for worker in workers
      for pos in gen_next_positions(worker)
    }

    def duplicate_positions(positions):
      for pos in positions:
        cell = self.game.game_map.get_cell_by_pos(pos)
        if cell.citytile is not None and cell.citytile.team == self.game.id:
          # TODO: inc
          for _ in range(4):
            yield pos
        else:
          yield pos

    next_positions = list(duplicate_positions(next_positions))

    # print(f'turn={g.turn}, compute_worker_moves next_positions={len(next_positions)}',
          # file=sys.stderr)
    def get_position_to_index():
      d = defaultdict(list)
      for i, pos in enumerate(next_positions):
        d[pos].append(i)
      return d

    position_to_indices = get_position_to_index()
    C = np.ones((len(workers), len(next_positions))) * -MAX_MOVE_WEIGHT
    for worker_idx, worker in enumerate(workers):
      shortest_path = self.shortet_paths[worker.id]

      shortest_path_points = None
      if worker.target_pos is not None:
        shortest_path_points = shortest_path.compute_shortest_path_points(worker.target_pos)
        # if DEBUG:
          # print(f'  w[{worker.id}]={worker.pos}, target={worker.target_pos}, S_path={shortest_path_points}', file=sys.stderr)

      for next_position in gen_next_positions(worker):
        wt = get_step_weight(worker, next_position,
                            shortest_path, shortest_path_points)
        C[worker_idx, position_to_indices[next_position]] = wt
        # if DEBUG:
        # print(f'w[{worker.id}]@{worker.pos} goto {next_position}, wt = {wt}', file=sys.stderr)

    # print(f'turn={g.turn}, compute_worker_moves before linear_sum_assignment',
          # file=sys.stderr)
    rows, cols = scipy.optimize.linear_sum_assignment(C, maximize=True)
    for worker_idx, poi_idx in zip(rows, cols):
      worker = workers[worker_idx]
      if not worker.can_act():
        continue

      wt = C[worker_idx, poi_idx]
      if wt < 0:
        print((f"w[{worker.id}]@{worker.pos}, next[{next_position}], "
              f"v={C[worker_idx, poi_idx]}, target={worker.target_pos}"),
              file=sys.stderr)
        assert wt >= 0

      next_position = next_positions[poi_idx]
      move_dir = worker.pos.direction_to(next_position)
      self.add_unit_action(worker, worker.move(move_dir))

      # print((f"w[{worker.id}]@{worker.pos}, next[{next_position}], "
             # f"v={C[worker_idx, poi_idx]}, target={worker.target_pos}"),
            # file=sys.stderr)


  def try_build_citytile(self):
    t = self.game.turn % CIRCLE_LENGH
    for unit in self.game.player.units:
      unit.is_building_city = False
      if not unit.is_worker():
        continue

      # Sitting on th cell of target position for city building.
      if (self.circle_turn < BUILD_CITYTILE_ROUND
          and unit.can_act()
          and unit.can_build(self.game.game_map)
          and (unit.target_pos and unit.target_pos == unit.pos
               or unit.target_cid > 0)):
        cell = self.game_map.get_cell_by_pos(unit.pos)
        # if cell.is_near_resource:
        self.add_unit_action(unit, unit.build_city())
        unit.is_building_city = True

  def compute_citytile_actions(self):
    player = self.game.player
    total_tile_count = player.city_tile_count
    total_unit_count = len(player.units)

    cur_research_points = player.research_points

    cities = list(player.cities.values())
    cities = sorted(cities, key=lambda c: get_city_no(c), reverse=True)
    action_citytile_positions = set()


    def every_citytiles(cities):
      for city in cities:
        for citytile in city.citytiles:
          if not citytile.can_act():
            continue
          yield citytile

    for citytile in every_citytiles(cities):
      if total_unit_count < total_tile_count and total_unit_count < MAX_UNIT_NUM:
        total_unit_count += 1
        self.actions.append(citytile.build_worker())
        action_citytile_positions.add(citytile.pos)

    cities = sorted(cities, key=lambda c: get_city_no(c))
    for citytile in every_citytiles(cities):
      if citytile.pos in action_citytile_positions:
        continue
      if cur_research_points < MAX_RESEARCH_POINTS:
        cur_research_points += 1
        self.actions.append(citytile.research())


  def compute_unit_shortest_paths(self):
    self.shortet_paths = {}

    empty_set = set()
    for unit in self.game.player.units:
      shortest_path = ShortestPath(self.game, unit.pos, empty_set)
      shortest_path.compute()
      self.shortet_paths[unit.id] = shortest_path

  def update_game_info(self):
    self.cluster_info = ClusterInfo(self.game)
    self.cluster_info.compute()

    self.update_game_map_unit_info()


  def assign_worker_to_resource_cluster(self):
    """For each no citytile cluster of size >= 2, find a worker with cargo space not 100.
      And set its target pos."""
    # Computes worker property. TODO move it out of this function.
    workers = [unit for unit in self.game.player.units
               if (unit.is_worker()
                   and unit.target_pos is None
                   and not unit.has_planned_action)]
    for i, worker in enumerate(workers):
      worker.cid_to_tile_pos = {}
      worker.cid_to_cluster_dist = {}

    if self.game.player.city_tile_count < 2:
      return

    if self.cluster_info.max_cluster_id == 0:
      return

    def worker_cluster_dist(worker, cid):
      x_pos, y_pos = np.where(self.cluster_info.position_to_cid == cid)
      n_resource_tile = len(x_pos)

      shortest_path = self.shortet_paths[worker.id]
      best_pos = None
      best_dist = MAX_PATH_WEIGHT
      for x, y in zip(x_pos, y_pos):
        pos = Position(x, y)
        dist = shortest_path.shortest_dist(pos)
        if best_dist <= dist:
          continue

        target_night_count = get_night_count_by_dist(self.game.turn, dist-1,
                                                     worker.cooldown,
                                                     get_unit_action_cooldown(worker))
        if worker.unit_night_count < target_night_count:
          continue
        best_dist = dist
        best_pos = pos
      return best_dist, best_pos, n_resource_tile

    def get_cluster_weight(worker, cid):
      fuel = self.cluster_info.cluster_fuel[cid]
      tile_dist, tile_pos, n_tile = worker_cluster_dist(worker, cid)
      if tile_dist >= MAX_PATH_WEIGHT:
        return -1
      # Save tile position
      worker.cid_to_tile_pos[cid] = tile_pos
      worker.cid_to_cluster_dist[cid] = tile_dist

      # if n_tile <= 1:
        # return 1
      wt = fuel / ((tile_dist / 5) + 1)
      return wt

    def gen_resource_clusters():
      for cid in range(self.cluster_info.max_cluster_id):
        x_pos, y_pos = np.where(self.cluster_info.position_to_cid == cid)
        n_resource_tile = len(x_pos)
        if n_resource_tile <= 1:
          continue
        yield cid

    resource_clusters = list(gen_resource_clusters())
    weights = np.ones((len(workers), len(resource_clusters))) * -1
    for j, cid in enumerate(resource_clusters):
      for i, worker in enumerate(workers):
        weights[i, j] = get_cluster_weight(worker, cid)


    # TODO: fine tune?
    MAX_EXPLORE_WORKE = 3
    explore_worker_num = 0

    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    for worker_idx, cluster_idx in zip(rows, cols):
      worker = workers[worker_idx]
      cid = resource_clusters[cluster_idx]
      if weights[worker_idx, cluster_idx] < 0:
        continue

      worker.target_cid = cid
      worker.target_cid_dist = worker.cid_to_cluster_dist[cid]

      # for debugging.
      tile_pos = worker.cid_to_tile_pos[cid]
      x = annotate.x(tile_pos.x, tile_pos.y)
      line = annotate.line(worker.pos.x, worker.pos.y, tile_pos.x, tile_pos.y)
      self.actions.extend([x, line])

      explore_worker_num += 1
      if explore_worker_num > MAX_EXPLORE_WORKE:
        break


  def worker_look_for_resource_transfer(self):
    circle_turn = self.circle_turn
    if circle_turn >= BUILD_CITYTILE_ROUND:
      return

    def cell_can_build_citytile(cell):
      return not cell.has_resource() and cell.citytile is None

    def neighbour_worker_with_enough_resource(worker):
      """Assign transfer action here, but leave movement to the existing process."""
      worker_amt = worker.get_cargo_space_left()
      nb_cells = get_neighbour_positions(worker.pos, self.game_map, return_cell=True)
      next_cells = nb_cells + [self.game_map.get_cell_by_pos(worker.pos)]

      # TODO: sort next cells
      for target_cell in next_cells:
        collect_amt, _ = get_one_step_collection_values(target_cell, self.game.player, self.game_map)
        if not cell_can_build_citytile(target_cell):
          continue

        # Worker can collect on its own.
        if worker_amt + collect_amt >= CITY_BUILD_COST:
          break

        # transfer + worker.action + worker.collect on next step
        for nb_cell in nb_cells:
          nb_unit = nb_cell.unit
          if (nb_unit is None
              or nb_unit.team != self.game.player.team
              or not nb_unit.can_act()
              or nb_unit.has_planned_action):
            continue

          # Neighbour can build.
          if (nb_unit.get_cargo_space_left() == 0
              and cell_can_build_citytile(nb_cell)):
            continue

          # TODO: support other resource: use max resource
          if (worker_amt + collect_amt + nb_unit.cargo.wood >= CITY_BUILD_COST):
            transfer_amount = CITY_BUILD_COST - (worker_amt + collect_amt)
            print(f'$A {worker.id}{worker.cargo} accept direct transfer from {nb_unit.id}{nb_unit.cargo} ${transfer_amount}')
            self.add_unit_action(nb_unit,
                                 nb_unit.transfer(worker.id, RESOURCE_TYPES.WOOD, transfer_amount))
            worker.transfer_build_locations.add(target_cell.pos)
            worker.is_transfer_worker = True

            # Assume the first worker transfer.
            break

            # TODO: compute move based on this cell.
            # self.add_unit_action(worker, worker.build_city())
            # return nb_unit


    # check direct transfer and build.
    # Pre-condition: worker.can_act & unit.can_act
    for worker in self.game.player.units:
      if worker.has_planned_action:
        continue
      if not worker.is_worker() or not worker.can_act():
        continue

      # worker already full.
      if worker.get_cargo_space_left() == 0:
        continue

      neighbour_worker_with_enough_resource(worker)

  def execute(self):
    actions = self.actions
    g = self.game
    player = g.player

    self.update_game_info()

    self.compute_unit_shortest_paths()

    self.compute_citytile_actions()

    self.worker_look_for_resource_transfer()

    self.assign_worker_to_resource_cluster()
    self.assign_worker_target()

    self.try_build_citytile()
    self.compute_worker_moves()


_strategy = Strategy()

def agent(observation, configuration):
  _strategy.update(observation, configuration)
  _strategy.execute()
  return _strategy.actions
