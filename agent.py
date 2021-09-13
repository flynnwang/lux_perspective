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

Total Matches: 224 | Matches Queued: 8
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
/Users/flynnwang/dev/playground/lux_perspective/main.py | 2kzojOcDHJ6N   | 25.1049731      | μ=27.890, σ=0.928  | 104
/Users/flynnwang/dev/playground/versions/explore_cluster/main.py | JGyLxWmBdbps   | 24.4152954      | μ=27.142, σ=0.909  | 116
/Users/flynnwang/dev/playground/versions/boost_near_resource/main.py | 21jnhfG9PXYJ   | 20.9467166      | μ=23.598, σ=0.884  | 116
/Users/flynnwang/dev/playground/versions/Tong_Hui_Kang/main.py | SRKabI4NF0S3   | 17.7132924      | μ=20.576, σ=0.954  | 112

opponent_name	Tong_Hui_Kang	boost_near_resource	explore_cluster	lux_perspective
Tong_Hui_Kang	0.000	36.842	8.889	18.750
boost_near_resource	65.789	0.000	36.585	22.500
explore_cluster	91.111	63.415	0.000	50.000
lux_perspective	81.250	77.500	52.941	0.000


* Assign maximum amount of resource worker per cluster.
* debug 437892541 at turn 49, worker going back.
* debug u_5 goto (6, 28) at turn 16: boost cluster weight not right.
* keep cluster assignment even it's conflict with citytile assignment.
* issue 1: why move out of city and wait.
* issue 2: why not stay in the city?
* do not move onto build action unit.


* Sorted transfer grid.
* Do not tranfer more times
* Do not pull cluster explore worker back for city building.
* Transfer worker not move onto transfer locations (across multiple turns)
* split different type cluster
* Move to coal and uranimu earlier (Estimate research points)
* dist=2 woker not moving back to city (because of round limit, >=28)
* [bugfix] map:371000968, initial round not moving


-> bugfix

->
* Save size 1 citytile
* Tune BUILD_CITYTILE_ROUND to 30 (for the last day, must build connect city)
* worker forward fuel (from coal and uranium) to city better?

* Move earlier into wood for city building. (at first hour of day?)
* Support other type of resource transfer.

* give multiple worker to coal & uranium cell?
* [minor opt]: do not move to cell has limit resource due to collection: predict cell dying





# TODO
* Add resource weight to city tile.
* Raise priority of near_resource_tile targeted worker

TODO:
* Add units (enemy and mine) to cell and check blocking
# TODO: build city away from near resource tile


- Learning

* unit cargo won't goto 100 at night
* in city tile, road level is 6 (which means, move to citytile at night is good)
* A full worker will not collect resource
"""
import functools
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
BUILD_CITYTILE_ROUND = 28

MAX_PATH_WEIGHT = 99999


MAX_UNIT_NUM = 60


def dist_decay(dist, game_map):
  if dist == 0:
    return 1
  return dist * UNIT_ACTION_COOLDOWN['WORKER']


def dist_to_days(dist):
  return dist * UNIT_ACTION_COOLDOWN['WORKER'] - 1


def worker_enough_cargo_to_build(worker, collected_amount):
  return (worker_total_cargo(worker) + collected_amount >= CITY_BUILD_COST)


@functools.lru_cache(maxsize=1024, typed=False)
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


def is_resource_researched(resource, player, move_days=0):
  plus = move_days * player.avg_research_point_growth
  if (resource.type == Constants.RESOURCE_TYPES.COAL
      and not player.researched_coal(plus)):
    return False
  if (resource.type == Constants.RESOURCE_TYPES.URANIUM
      and not player.researched_uranium(plus)):
    return False
  return True


def get_cell_resource_values(cell, player, unit=None, move_days=0):
  # (amount, fuel)
  if not cell.has_resource():
    return 0, 0

  resource = cell.resource
  if not is_resource_researched(resource, player, move_days):
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
  for nb_cell in get_neighbour_positions(cell.pos, game_map, return_cell=True):
    nights += estimate_resource_night_count(nb_cell.resource, upkeep)
  return nights


def get_one_step_collection_values(cell, player, game_map, move_days=0):
  amount, fuel = get_cell_resource_values(cell, player, move_days=move_days)
  for nb_cell in get_neighbour_positions(cell.pos, game_map, return_cell=True):
    a, f = get_cell_resource_values(nb_cell, player, move_days=move_days)
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
      for nb_cell in get_neighbour_positions(cur, self.game_map, return_cell=True):
        if cell_has_opponent_citytile(nb_cell, self.game):
          continue

        # Skip player unit in cooldown.
        if (nb_cell.unit and
            nb_cell.unit.team == self.game.opponent_id and not nb_cell.unit.can_act()):
          continue

        newpos = nb_cell.pos
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
      for nb_cell in get_neighbour_positions(cur, self.game_map, return_cell=True):
        newpos = nb_cell.pos
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
      for nb_cell in get_neighbour_positions(cur, self.game_map, return_cell=True):
        newpos = nb_cell.pos
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
      # return (cell.has_resource()
              # and is_resource_researched(cell.resource, self.game.player))
      return (cell.has_resource())

    def search_cluster(start_cell):
      vistied_citytile_positions = set()
      total_fuel = resource_fuel(start_cell.resource)
      q = deque([start_cell.pos])
      self.set_cid(start_cell.pos, max_cluster_id)

      while q:
        cur = q.popleft()
        cur_cell = self.game_map.get_cell_by_pos(cur)
        for nb_cell in get_neighbour_positions(cur, self.game_map, return_cell=True):
          newpos = nb_cell.pos

          # Count citytile.
          # if (nb_cell.citytile != None
              # and nb_cell.citytile.team == self.game.player.team):
          if (nb_cell.citytile != None):
            vistied_citytile_positions.add(nb_cell.citytile.pos)

          if not is_valid_resource_cell(nb_cell):
            continue

          # Split different resource into different cluster
          if cur_cell.resource.type != nb_cell.resource.type:
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
    self.accepted_transfer_offers = {}

  @property
  def circle_turn(self):
    return self.game.turn % CIRCLE_LENGH

  @property
  def days_this_round(self):
    return DAY_LENGTH - self.circle_turn

  @property
  def game_map(self):
    return self.game.game_map

  @property
  def player(self):
    return self.game.player

  def player_available_workers(self):
    workers = [unit for unit in self.game.player.units
               if (unit.is_worker()
                   and unit.target_pos is None
                   and not unit.has_planned_action)]
    return workers

  def update(self, observation, configuration):
    self.game.update(observation, configuration)

    # Clear up actions for current step.
    self.actions = []

  def add_unit_action(self, unit, action):
    assert unit.has_planned_action == False

    unit.has_planned_action = True
    self.actions.append(action)



  def update_game_map_info(self):
    for y in range(self.game.map_height):
      for x in range(self.game.map_width):
        cell = self.game_map.get_cell(x, y)
        cell.unit = None
        cell.is_near_resource = self.is_near_resource_cell(cell)
        cell.is_citytile_neighbour = self.has_citytile_neighbours(cell)
        cell.has_buildable_neighbour = False

    for unit in self.game.player.units:
      cell = self.game_map.get_cell_by_pos(unit.pos)
      cell.unit = unit

    for unit in self.game.opponent.units:
      cell = self.game_map.get_cell_by_pos(unit.pos)
      cell.unit = unit

  def update_unit_info(self):
    self.shortet_paths = {}
    empty_set = set()
    for unit in self.game.player.units:
      unit.has_planned_action = False
      unit.target_pos = None

      unit.transfer_build_locations = set()
      unit.target_cluster_id = -1
      unit.is_transfer_worker = False
      unit.target_city_id = None

      unit.unit_night_count = cargo_night_endurance(unit.cargo, get_unit_upkeep(unit))

      round_night_count = get_night_count_this_round(self.game.turn)
      unit.is_dying = unit.unit_night_count < round_night_count
      unit.cell = self.game_map.get_cell_by_pos(unit.pos)

      # Shortest path
      shortest_path = ShortestPath(self.game, unit.pos, empty_set)
      shortest_path.compute()
      self.shortet_paths[unit.id] = shortest_path

      unit.cid_to_tile_pos = {}
      unit.cid_to_cluster_dist = {}


  def has_citytile_neighbours(self, cell):
    for nb_cell in get_neighbour_positions(cell.pos, self.game_map, return_cell=True):
      if cell_has_player_citytile(nb_cell, self.game):
        return True
    return False

  def is_near_resource_cell(self, cell):
    def cell_near_done_research_resource(cell):
      for nb_cell in get_neighbour_positions(cell.pos, self.game_map, return_cell=True):
        if (nb_cell.has_resource()
            and is_resource_researched(nb_cell.resource, self.game.player)):
          return True
      return False

    return (not cell.has_resource()
            and cell.citytile is None
            and cell_near_done_research_resource(cell))

  def assign_worker_target(self):
    MAX_UNIT_PER_CITY = 4
    g = self.game
    player = g.player

    workers = [unit for unit in player.units
               if (unit.is_worker()
                   and unit.target_pos is None
                   and not unit.has_planned_action)]

    def cell_has_buildable_neighbour(cell):
      for nb_cell in get_neighbour_positions(cell.pos, self.game_map, return_cell=True):
        if (not nb_cell.has_resource()
            and nb_cell.citytile is None):
          return True
      return False


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
        elif cell.is_near_resource:
          is_target_cell = True
        elif cell.is_citytile_neighbour:
          is_target_cell = True
        elif (cell.unit and cell.unit.team == player.team):
          is_target_cell = True

        if is_target_cell:
          cell.has_buildable_neighbour = cell_has_buildable_neighbour(cell)
          target_cells.append(cell)

    def is_resource_tile_can_save_dying_worker(resource_tile, worker, dist):
      if (not resource_tile.has_resource()
          or (not is_resource_researched(resource_tile.resource, player,
                                         move_days=dist_to_days(dist)))):
        return False
      if worker.is_dying:
        cell_nights = estimate_cell_night_count(cell, get_unit_upkeep(worker), g.game_map)
        round_nights = get_night_count_this_round(g.turn)
        if worker.unit_night_count + cell_nights >= round_nights:
          return True
      return False

    MAX_WEIGHT_VALUE = 10000
    CLUSTER_BOOST_WEIGHT = 200 * 40
    UNIT_SAVED_BY_RES_WEIGHT = 10
    def get_resource_weight(worker, resource_tile, dist):
      # Use dist - 1, because then the worker will get resource.
      target_night_count = get_night_count_by_dist(g.turn, dist-1, worker.cooldown,
                                                   get_unit_action_cooldown(worker))
      # TODO: add some buffer for safe arrival
      if worker.unit_night_count < target_night_count:
        return -1

      # Give a small weight for any resource 0.1 TODO: any other option?
      wt = 0

      # For wood cell with no buildable neightbor, demote its weight
      if (is_resource_wood(resource_tile.resource)
          and not resource_tile.has_buildable_neighbour):
        # TODO: use dist * 2 for estimate of number of days to cell
        _, fuel = get_cell_resource_values(resource_tile, player,
                                           move_days=dist_to_days(dist))
        fuel /= 2
        # if resource_tile.pos == Position(5, 28):
          # print(f'triggered demote for no neighbour c[{resource_tile.pos}]')
      else:
        _, fuel = get_one_step_collection_values(resource_tile, player, g.game_map,
                                                 move_days=dist_to_days(dist))
      wt += fuel

      # Try to hide next to resource grid.
      if is_resource_tile_can_save_dying_worker(resource_tile, worker, dist):
        wt += UNIT_SAVED_BY_RES_WEIGHT


      # TODO: Consider drop cluster boosting when dist <= 1
      cid = self.cluster_info.get_cid(resource_tile.pos)
      boost_cluster = 0
      if (worker.target_cluster_id >= 0 and worker.target_cluster_id == cid):
          # worker.cid_to_cluster_dist[cid] > 1):
        cluster_fuel_factor = self.cluster_info.query_cluster_fuel_factor(resource_tile.pos)
        boost_cluster += CLUSTER_BOOST_WEIGHT * cluster_fuel_factor

      # if worker.id == 'u_8' and resource_tile.pos in [Position(5, 24), Position(6, 23)]:
        # print(f"t={g.turn},cell={resource_tile.pos}, wt={wt}, wt_c={boost_cluster}")

      return wt / dist_decay(dist, g.game_map) + boost_cluster
      # return wt / (dist + 0.1) + boost_cluster
      # return wt / dist_decay(dist, g.game_map)

    def get_city_tile_weight(worker, city_cell, dist):
      """
      1. collect fuel
      2. protect dying worker [at night]
      3. receive fuel from a fuel full worker on cell
      """
      CITYTILE_LOST_WEIGHT = 200

      # TODO: It's asuming dist are full of danger, but it could be move inside citytile.
      target_night_count = get_night_count_by_dist(g.turn, dist, worker.cooldown,
                                                   get_unit_action_cooldown(worker))
      if worker.unit_night_count < target_night_count:
        return -1

      citytile = city_cell.citytile
      city = g.player.cities[citytile.cityid]

      # Stay at city will gain this amout of fuel
      amount, fuel = get_one_step_collection_values(citytile.cell, player, g.game_map)
      wt = 0
      if self.game.is_night:
        wt += max(fuel - LIGHT_UPKEEP['CITY'], 0)

      # Try to hide in the city if worker will run out of fuel at night
      city_will_last = not city_wont_last_at_nights(g.turn, city)
      _, worker_collect_fuel = get_cell_resource_values(worker.cell, player)
      is_safe_cell = (worker_collect_fuel >= get_unit_upkeep(worker))
      boost_dying_worker = False
      if (worker.is_dying and city_will_last and not is_safe_cell):
        wt += 1
        boost_dying_worker = True


      # if worker.id == 'u_6':
        # print(f"w[{worker.id}], t[{citytile.pos}], turn_c={self.circle_turn >= BUILD_CITYTILE_ROUND}, dying={worker.is_dying}, city_last={city_will_last}", file=sys.stderr)

      # TODO(wangfei): estimate worker will full
      n_citytile = len(city.citytiles)


      # Try move onto city to save it
      min_fuel = (self.game.is_day and 60 or 60)
      min_city_last = 20 if g.turn < 320 else 10
      # TODO: is this round needed? or could be fine turned
      if (self.circle_turn >= 20
          and n_citytile >= 2
          and city_last_nights(city) < min_city_last
          and worker_total_fuel(worker) >= min_fuel):
        wt += worker_total_fuel(worker)
        if not city_will_last:
          wt += worker_total_fuel(worker)

      # Boost when worker has resource and city tile won't last.
      days_left = DAY_LENGTH - self.circle_turn - worker.cooldown
      round_nights = get_night_count_this_round(g.turn)

      # Boost based on woker, city assignment. (not used)
      if (worker.target_city_id == citytile.cityid
          and worker.pos.distance_to(city_cell.pos) == 1):
        wt += 1000 * n_citytile

        # More weights on dying city
        # if city_left_days < round_nights:
          # wt += n_citytile * CITYTILE_LOST_WEIGHT

      # When there is no fuel on the map, go back to city
      if self.cluster_info.max_cluster_fuel == 0:
        wt += 100

      # TODO: support night
      # Save more citytile if worker has enough resource to save it
      # - enough time to arrive
      # - with substential improvement of its living
      # TODO(wangfei): try delete this rule
      unit_time_cost = (dist - 1) * get_unit_action_cooldown(worker) + 1
      city_left_days = math.floor(city.fuel / city.light_upkeep)
      woker_fuel = worker_total_fuel(worker)
      city_left_days_deposited = math.floor((city.fuel + woker_fuel) / city.light_upkeep)
      if (days_left >= unit_time_cost
          and city_left_days < NIGHT_LENGTH
          and city_left_days_deposited > NIGHT_LENGTH
          and city_left_days_deposited - city_left_days >= (18 // n_citytile)):
        wt += CITYTILE_LOST_WEIGHT * len(city.citytiles)

      # if worker.id == 'u_7' and city.id in ['c_12']:
        # print(f"t={g.turn}, {worker.id}, nc={worker.unit_night_count}, tnc={target_night_count} tile={citytile.pos}, wt={wt}, dying={worker.is_dying}, boost_dying={boost_dying_worker}")
      return wt / dist_decay(dist, g.game_map)
      # return wt / (dist + 0.1)

    def cell_next_to_target_cluster(worker, near_resource_tile):
      if worker.target_cluster_id < 0:
        return False

      for nb_cell in get_neighbour_positions(near_resource_tile.pos, g.game_map, return_cell=True):
        newpos = nb_cell.pos
        cid = self.cluster_info.get_cid(newpos)
        if cid == worker.target_cluster_id:
          return True
      return False

      # TODO(wangfei): merge near resource tile and resource tile weight functon
    def get_near_resource_tile_weight(worker, near_resource_tile, dist):
      target_night_count = get_night_count_by_dist(g.turn, dist, worker.cooldown,
                                                   get_unit_action_cooldown(worker))
      # TODO: add some buffer for safe arri0
      if worker.unit_night_count < target_night_count:
        return -1

      amount, fuel = get_one_step_collection_values(near_resource_tile, player,
                                                    g.game_map, move_days=dist_to_days(dist))
      wt = 0.1 + fuel

      # Boost the target collect amount by 2 (for cooldown) to test for citytile building.
      # it's 2, because one step move and one step for cooldown
      build_city_bonus = False
      days_left = BUILD_CITYTILE_ROUND - self.circle_turn - worker.cooldown
      if (self.circle_turn < BUILD_CITYTILE_ROUND
          and worker_enough_cargo_to_build(worker, amount*2)
          and days_left >= (dist - 1) * get_unit_action_cooldown(worker) + 1):
        build_city_bonus = True

      # TODO: maybe time consuming
      if (dist <= 4 and
          self.shortet_paths[worker.id].path_blocked_by_citytile(near_resource_tile.pos)):
        build_city_bonus = False

      boost_cluster = 0
      if cell_next_to_target_cluster(worker, near_resource_tile):
        cid = worker.target_cluster_id
        cluster_fuel_factor = self.cluster_info.cluster_fuel[cid] / self.cluster_info.max_cluster_fuel
        boost_cluster += CLUSTER_BOOST_WEIGHT * cluster_fuel_factor

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

      p = near_resource_tile.pos

      # Try to hide next to resource grid.
      if is_resource_tile_can_save_dying_worker(near_resource_tile, worker, dist):
        wt += UNIT_SAVED_BY_RES_WEIGHT

      # if worker.id == 'u_3' and near_resource_tile.pos in [Position(5, 28), Position(8, 24),
                                                           # Position(4, 27)]:
        # print(f"t={g.turn}, near_resource_cell={near_resource_tile.pos}, dist={dist}, wt={wt}, wt_c={boost_cluster}, amt={amount}, fuel={fuel}")
      return wt / dist_decay(dist, g.game_map) + boost_cluster
      # return wt / (dist + 0.1) + boost_cluster

    def get_city_tile_neighbour_weight(worker, cell):
      wt = -999
      # If only do one transfer, then need to sort the positions
      if cell.pos in worker.transfer_build_locations:
        nb_citytiles = [nb_cell.citytile
                        for nb_cell in get_neighbour_positions(cell.pos, self.game_map, return_cell=True)
                        if cell_has_player_citytile(nb_cell, self.game)]
        assert len(nb_citytiles) > 0
        wt = 300 * len(nb_citytiles)

      # Stick worker onto citytile neighoubr to build city tile
      offer = self.accepted_transfer_offers.get(worker.id)
      if offer:
        offer_turn, offer_pos = offer
        if (self.game.turn - offer_turn <= 2
            and offer_pos == worker.pos
            and worker.pos == cell.pos):
          wt = MAX_WEIGHT_VALUE
          # print(f'BBBB {worker.id}@[{worker.pos}], cell={cell.pos}')

      # if worker.id == 'u_7':
        # print(f'!!! {worker.id}, transfer_pos_last_round={offer}')
      return wt

    def get_worker_tile_weight(worker, target):
      wt = -99999
      if worker.pos == target.pos:
        wt = 0.1
      return wt


    # Value matrix for ship target assginment
    # * row: workers
    # * column: point of interests
    weights = np.ones((len(workers), len(target_cells))) * -9999

    # MAIN_PRINT
    print((f'*turn={g.turn}, #W={len(workers)}, #C={player.city_tile_count} '
           f'R={g.player.research_points}'),
          file=sys.stderr)
    for i, worker in enumerate(workers):
      for j, target in enumerate(target_cells):
        # Can't arrive at target cell.
        shortest_path = self.shortet_paths[worker.id]
        dist = shortest_path.shortest_dist(target.pos)
        if dist >= MAX_PATH_WEIGHT:
          continue

        v = 0
        if cell_has_player_citytile(target, self.game):
          v = get_city_tile_weight(worker, target, dist)
        elif target.has_resource():
          v = get_resource_weight(worker, target, dist)
        elif target.is_near_resource:
          v = get_near_resource_tile_weight(worker, target, dist)
        elif target.is_citytile_neighbour:
          v = get_city_tile_neighbour_weight(worker, target)
        else:
          v = get_worker_tile_weight(worker, target)
        weights[i, j] = v

        # if worker.id == 'u_6' and target.pos in [Position(8, 25), Position(6, 25),]:
          # print(f"w[{worker.id}], cd={worker.cooldown}, t[{target.pos}], wt={v:.1f}", file=sys.stderr)



    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    for worker_idx, target_idx in zip(rows, cols):
      worker = workers[worker_idx]
      target = target_cells[target_idx]
      worker.target_pos = target.pos
      if weights[worker_idx, target_idx] < 0:
        continue

      if DEBUG:
        # print(f'{worker.id}, v={weights[worker_idx, target_idx]}', file=sys.stderr)
        # x = annotate.x(worker.pos.x, worker.pos.y)
        c = annotate.circle(target.pos.x, target.pos.y)
        # if worker.target_cluster_id >= 0:
          # c = annotate.x(target.pos.x, target.pos.y)

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
      return [worker.pos] + [c.pos for c in get_neighbour_positions(worker.pos, g.game_map, return_cell=True)]

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
      # Sit and build.
      if (self.circle_turn < BUILD_CITYTILE_ROUND
          and unit.can_act()
          and unit.can_build(self.game.game_map)
          and (unit.target_pos and unit.target_pos == unit.pos
               or unit.target_cluster_id >= 0)):
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



  def update_player_info(self):
    # Estimate number of research point in left day times.
    n_city = len(self.player.cities)
    self.player.avg_research_point_growth = n_city / CITY_ACTION_COOLDOWN

  def update_game_info(self):
    self.cluster_info = ClusterInfo(self.game)
    self.cluster_info.compute()

    self.update_player_info()
    self.update_game_map_info()
    self.update_unit_info()


  def assign_worker_to_resource_cluster(self):
    """For each no citytile cluster of size >= 2, find a worker with cargo space not 100.
      And set its target pos."""
    # Computes worker property.
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

      # if self.game.turn == 15:
        # print(f"{worker.id}, target_tile[{best_pos}, dist={best_dist}, unit_nights={worker.unit_night_count}], target_night_count={target_night_count}",
            # file=sys.stderr)
      return best_dist, best_pos, n_resource_tile

    def get_cluster_weight(worker, cid):
      # TODO: MAYBE keep it?
      if worker.target_city_id:
        return -9999

      fuel = self.cluster_info.cluster_fuel[cid]
      tile_dist, tile_pos, n_tile = worker_cluster_dist(worker, cid)
      if tile_dist >= MAX_PATH_WEIGHT:
        return -9999

      # resource cluster not researched.
      cell = self.game_map.get_cell_by_pos(tile_pos)
      if not is_resource_researched(cell.resource, self.player,
                                    move_days=dist_to_days(tile_dist)):
        return -9999

      # Save tile position
      worker.cid_to_tile_pos[cid] = tile_pos
      worker.cid_to_cluster_dist[cid] = tile_dist

      # if n_tile <= 1:
        # return 1
      # wt = fuel / ((tile_dist / 5) + 1)
      wt = fuel / dist_decay(tile_dist, self.game_map)
      return wt

    def gen_resource_clusters():
      for cid in range(self.cluster_info.max_cluster_id):
        x_pos, y_pos = np.where(self.cluster_info.position_to_cid == cid)
        n_resource_tile = len(x_pos)
        if n_resource_tile <= 1:
          continue
        yield cid

    workers = self.player_available_workers()
    resource_clusters = list(gen_resource_clusters())
    weights = np.ones((len(workers), len(resource_clusters))) * -9999
    for j, cid in enumerate(resource_clusters):
      for i, worker in enumerate(workers):
        weights[i, j] = get_cluster_weight(worker, cid)


    MAX_EXPLORE_WORKE = 3
    explore_worker_num = 0
    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    sorted_pairs = sorted(list(zip(rows, cols)), key=lambda x: -weights[x[0], x[1]])
    for worker_idx, cluster_idx in sorted_pairs:
      worker = workers[worker_idx]
      cid = resource_clusters[cluster_idx]
      if weights[worker_idx, cluster_idx] < 0:
        continue

      worker.target_cluster_id = cid
      worker.target_cid_dist = worker.cid_to_cluster_dist[cid]

      # for debugging.
      tile_pos = worker.cid_to_tile_pos[cid]
      x = annotate.x(tile_pos.x, tile_pos.y)
      line = annotate.line(worker.pos.x, worker.pos.y, tile_pos.x, tile_pos.y)
      self.actions.extend([x, line])

      # print(f'Assign Cluster {worker.id}, cell[{tile_pos}], wt={weights[worker_idx, cluster_idx]}')

      explore_worker_num += 1
      if explore_worker_num >= MAX_EXPLORE_WORKE:

        break


  def worker_look_for_resource_transfer(self):
    circle_turn = self.circle_turn
    if circle_turn >= BUILD_CITYTILE_ROUND:
      return

    def cell_can_build_citytile(cell):
      is_resource_tile = cell.has_resource()
      is_citytile = cell.citytile is not None
      unit_can_not_act = (cell.unit and not cell.unit.can_act())
      return (not is_resource_tile
              and not is_citytile
              and not unit_can_not_act)

    def neighbour_cells_can_build_citytile(cell):
      for nb_cell in get_neighbour_positions(cell.pos, self.game_map, return_cell=True):
        if cell_can_build_citytile(nb_cell):
          return True
      return False

    def get_neighbour_cell_weight(cell):
      wt = 0
      for nb_cell in get_neighbour_positions(cell.pos, self.game_map, return_cell=True):
        if cell_has_player_citytile(nb_cell, self.game):
          wt += 100
        elif nb_cell.is_near_resource:
          wt += 1000
      return wt

    def neighbour_worker_with_enough_resource(worker):
      """Assign transfer action here, but leave movement to the existing process."""
      worker_amt = worker_total_cargo(worker)
      nb_cells = get_neighbour_positions(worker.pos, self.game_map, return_cell=True)
      next_cells = nb_cells + [self.game_map.get_cell_by_pos(worker.pos)]

      # Sort to prioritize better cells.
      next_cells = sorted(next_cells, key=lambda c: -get_neighbour_cell_weight(c))
      for target_cell in next_cells:
        # Target cell must can build citytile
        if not cell_can_build_citytile(target_cell):
          continue

        # TODO: try make one step collections value more accurate
        # Worker can collect on its own.
        collect_amt, _ = get_one_step_collection_values(target_cell, self.game.player, self.game_map)
        # collect_amt = 0
        if worker_amt + collect_amt >= CITY_BUILD_COST:
        # collect_amt /= 2  # Discount this value
        # if worker_amt + collect_amt >= CITY_BUILD_COST:
        # if worker_amt >= CITY_BUILD_COST:
          break

        # transfer + worker.action + worker.collect on next step
        for nb_cell in nb_cells:
          nb_unit = nb_cell.unit
          # Neighbour is not valid or ready.
          if (nb_unit is None
              or nb_unit.team != self.game.player.team
              or not nb_unit.can_act()
              or nb_unit.has_planned_action):
            continue

          # Neighbour can build on its own cell.
          if (nb_unit.get_cargo_space_left() == 0
              and (cell_can_build_citytile(nb_cell)
                   or neighbour_cells_can_build_citytile(nb_cell))):
            continue


          # TODO: support other resource: use max resource
          if (worker_amt + collect_amt + nb_unit.cargo.wood >= CITY_BUILD_COST):
            transfer_amount = CITY_BUILD_COST - (worker_amt + collect_amt)
            # transfer_amount = CITY_BUILD_COST - (worker_amt)
            # transfer_amount = nb_unit.cargo.wood
            print(f'$A {worker.id}{worker.cargo}@{worker.pos} accept transfer from {nb_unit.id}{nb_unit.cargo} ${transfer_amount} to goto {target_cell.pos}')
            self.add_unit_action(nb_unit,
                                 nb_unit.transfer(worker.id, RESOURCE_TYPES.WOOD, transfer_amount))
            worker.transfer_build_locations.add(target_cell.pos)
            worker.is_transfer_worker = True

            # TODO: maybe add worker action directly here?
            self.accepted_transfer_offers[worker.id] = (self.game.turn, target_cell.pos)

            # Assume the first worker transfer.
            return


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


  def assign_worker_city_target(self):
    MAX_CITY_ARRIVAL = 2

    if self.circle_turn < BUILD_CITYTILE_ROUND:
      return

    def worker_neighbour_cities(worker):
      for nb_cell in get_neighbour_positions(worker.pos, self.game_map,
                                          return_cell=True):
        if cell_has_player_citytile(nb_cell, self.game):
          yield nb_cell.citytile.cityid


    MIN_CITY_WEIGHT = -9999
    def get_city_weight(worker, city):
      # If current worker won't help, skip it.
      if not worker.can_act():
        return MIN_CITY_WEIGHT

      # There is a cell for worker to go back and save itself.
      _, worker_collect_fuel = get_cell_resource_values(worker.cell, self.player)
      is_safe_cell = (worker_collect_fuel >= get_unit_upkeep(worker))
      if not is_safe_cell:
        return MIN_CITY_WEIGHT

      # Do not go to city if city can wait and worker is not full enough.
      city_left_days = math.floor(city.fuel / city.light_upkeep)
      worker_cargo_amt = worker_total_cargo(worker)
      if city_left_days > 0 and worker_cargo_amt < 95:
        return MIN_CITY_WEIGHT

      worker_fuel = worker_total_fuel(worker)
      city_left_days_deposited = math.floor((city.fuel + worker_fuel)
                                           / city.light_upkeep)
      n_citytile = len(city.citytiles)
      delta_days = city_left_days_deposited - city_left_days
      wt = n_citytile * delta_days
      if city_left_days == 0 and delta_days > 0:
        wt += 100
      return wt

    workers = self.player_available_workers()
    cities = list(self.player.cities.values()) * MAX_CITY_ARRIVAL
    weights = np.ones((len(workers), len(cities))) * MIN_CITY_WEIGHT
    for i, worker in enumerate(workers):
      neighbour_city_ids = set(worker_neighbour_cities(worker))

      for j, city in enumerate(cities):
        if city.cityid not in neighbour_city_ids:
          continue
        weights[i, j] = get_city_weight(worker, city)

    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    for worker_idx, city_idx in zip(rows, cols):
      worker = workers[worker_idx]
      city = cities[city_idx]
      if weights[worker_idx, city_idx] < 0:
        continue

      worker.target_city_id = city.cityid
      print(f'{worker.id} is sending to {city.cityid}')


  def execute(self):
    actions = self.actions
    g = self.game
    player = g.player

    self.update_game_info()

    self.compute_citytile_actions()

    # self.assign_worker_city_target()
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
