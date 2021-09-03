"""
* limit research point to 200 and build worker with new citytiles.
* try strategy Toad
* Only focus on near resource tile for building


Total Matches: 131 | Matches Queued: 19
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
/Users/flynnwang/dev/playground/versions/fast_city_far_resource/main.py | 1eJKYDLxJr9B   | 24.1486851      | μ=26.726, σ=0.859  | 131
/Users/flynnwang/dev/playground/lux_perspective/main.py | 4UwG826nuJ9d   | 20.6959079      | μ=23.274, σ=0.859  | 131


* Use collect amount to define resource weight
* Skip citytile during night

Total Matches: 22 | Matches Queued: 18
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
/Users/flynnwang/dev/playground/versions/fast_city_far_resource/main.py | QCSlghFk3qpd   | 21.2978680      | μ=28.406, σ=2.369  | 22
/Users/flynnwang/dev/playground/lux_perspective/main.py | OOC4EmHMBWW0   | 14.4867675      | μ=21.594, σ=2.369  | 22


* use collect amount X fuel rate, revert night forbidden_positions.

Total Matches: 41 | Matches Queued: 19
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
/Users/flynnwang/dev/playground/versions/fast_city_far_resource/main.py | HCmof1oX6QfD   | 23.0505634      | μ=27.061, σ=1.337  | 41
/Users/flynnwang/dev/playground/lux_perspective/main.py | 59h0nB9ntfen   | 18.9291842      | μ=22.939, σ=1.337  | 41


* revert night weights

Total Matches: 44 | Matches Queued: 20
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
/Users/flynnwang/dev/playground/versions/fast_city_far_resource/main.py | XA6yt0G6RDpw   | 22.9055557      | μ=26.636, σ=1.244  | 44
/Users/flynnwang/dev/playground/lux_perspective/main.py | hhED6DV4wMVJ   | 19.6328387      | μ=23.364, σ=1.244  | 44


* Move to city that will last at night

Total Matches: 109 | Matches Queued: 19
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
/Users/flynnwang/dev/playground/versions/fast_city_far_resource/main.py | j3bqdqRms2Et   | 23.4153534      | μ=26.061, σ=0.882  | 109
/Users/flynnwang/dev/playground/lux_perspective/main.py | a4ymKWS0ZTjP   | 21.2940286      | μ=23.939, σ=0.882  | 109


* Use near resource tile cell value with worker cargo to boost its weight

Tournament - ID: roq9bF, Name: Lux AI Season 1 Tournament | Dimension - ID: GMvRXx, Name: Lux
Status: running | Competitors: 2 | Rank System: trueskill

Total Matches: 64 | Matches Queued: 20
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
/Users/flynnwang/dev/playground/versions/fast_city_far_resource/main.py | 83Rj0D3ZYg8A   | 22.3923900      | μ=25.528, σ=1.045  | 64
/Users/flynnwang/dev/playground/lux_perspective/main.py | 2ewpYlZUbSVn   | 21.3372860      | μ=24.472, σ=1.045  | 64



# TODO: build city away from near resource tile


TODO:
* Add units (enemy and mine) to cell and check blocking
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

BUILD_CITYTILE_ROUND = 26

MAX_PATH_WEIGHT = 99999


def dist_decay(dist, game_map):
  # decay = game_map.width
  decay = 1
  return (dist / decay) + 1

def is_within_map_range(pos, game_map):
  return 0 <= pos.x < game_map.width and 0 <= pos.y < game_map.height

def worker_total_cargo(worker):
  cargo = worker.cargo
  return (cargo.wood + cargo.coal + cargo.uranium)

def worker_enough_cargo_to_build(worker, near_resource_tile_value):
  return worker_total_cargo(worker) + near_resource_tile_value >= CITY_BUILD_COST


def get_neighbour_positions(pos, game_map):
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
  return citytile is not None and citytile.team == game.opponent_id


def is_cell_done_research(cell, player):
  resource = cell.resource
  if (resource.type == Constants.RESOURCE_TYPES.COAL
      and not player.researched_coal()):
    return False
  if (resource.type == Constants.RESOURCE_TYPES.URANIUM
      and not player.researched_uranium()):
    return False
  return True

def get_cell_resource_value(cell, player):
  if not cell.has_resource():
    return 0

  if not is_cell_done_research(cell, player):
    return 0

  resource = cell.resource
  collection_rate = get_worker_collection_rate(resource)
  fuel_rate = get_resource_to_fuel_rate(resource)
  return min(collection_rate, resource.amount) * fuel_rate


def mark_boundary_resource(resource_tiles, game_map):
  for cell in resource_tiles:
    cell.is_boundary_resource = False
    for pos in get_neighbour_positions(cell.pos, game_map):
      nb_cell = game_map.get_cell_by_pos(pos)
      if not nb_cell.has_resource() and nb_cell.citytile is None:
        cell.is_boundary_resource = True


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

  def update(self, observation, configuration):
    game_state = self

    ### Do not edit ###
    if observation["step"] == 0:
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])


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
    # return

    total_append = 0
    while q:
      cur = q.popleft()
      cur_dist = self.dist[cur.x, cur.y]
      for newpos in get_neighbour_positions(cur, self.game_map):
        nb_cell = self.game_map.get_cell_by_pos(newpos)

        if cell_has_opponent_citytile(nb_cell, self.game):
          continue

        # Skip player unit in cooldown.
        if hasattr(nb_cell, 'unit') and not nb_cell.unit.can_act():
          continue
        if newpos in self.forbidden_positions:
          continue

        nb_dist = self.dist[newpos.x, newpos.y]
        if cur_dist + 1 >= nb_dist:
          continue

        self.dist[newpos.x, newpos.y] = cur_dist + 1
        q.append(newpos)
        total_append += 1
        # print(f' start_from {self.start_pos}, append {newpos}, cur_dist={cur_dist}', file=sys.stderr)

    # print(f'compute for pos {self.start_pos}, totol_append={total_append}', file=sys.stderr)

  def shortest_dist(self, pos):
    return self.dist[pos.x, pos.y]

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


class Strategy:

  def __init__(self):
    self.actions = []
    self.game = LuxGame()
    self.citytile_positions = set()

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

  def add_unit_action(self, unit, action):
    assert unit.has_planned_action == False

    unit.has_planned_action = True
    self.actions.append(action)


  def cell_near_done_research_resource(self, cell):
    for pos in get_neighbour_positions(cell.pos, self.game_map):
      nb_cell = self.game_map.get_cell_by_pos(pos)
      if (nb_cell.has_resource()
          and is_cell_done_research(nb_cell, self.game.player)):
        return True
    return False


  def update_wood_citytile_positions(self, player_city_tiles):
    self.citytile_positions = set()
    for citytile in player_city_tiles:
      self.citytile_positions.add(citytile.pos)

    for unit in self.game.player.units:
      cell = self.game_map.get_cell_by_pos(unit.pos)
      cell.unit = unit

    for unit in self.game.opponent.units:
      cell = self.game_map.get_cell_by_pos(unit.pos)
      cell.unit = unit


  def assign_worker_target(self):
    MAX_UNIT_PER_CITY = 4
    g = self.game
    player = g.player

    workers = [unit for unit in player.units if unit.is_worker()]

    # TODO: remove unit occupied by opponent_unit and opponent_citytile
    resource_tiles: list[Cell] = []
    citytiles = []
    near_resource_tiles = []
    worker_tiles = []
    citytile_count = 0
    for y in range(g.map_height):
      for x in range(g.map_width):
        cell = g.game_map.get_cell(x, y)
        cell.is_near_resource = False

        if cell.citytile and cell.citytile.team == player.team:
          citytiles.append(cell.citytile)
          continue

        if cell.has_resource():
          resource_tiles.append(cell)
          continue

        if (not cell.has_resource()
            and cell.citytile is None
            and self.cell_near_done_research_resource(cell)):
          near_resource_tiles.append(cell)
          cell.is_near_resource = True
          continue

        if hasattr(cell, 'unit') and cell.unit.team == player.team:
          worker_tiles.append(cell)
          continue
        # TODO: add dist 2 neighbour

    mark_boundary_resource(resource_tiles, g.game_map)

    assert len(citytiles) == len(g.player_city_tiles)
    n_citytile = len(citytiles)
    citytiles = citytiles * MAX_UNIT_PER_CITY

    targets = resource_tiles + citytiles + near_resource_tiles + worker_tiles

    def is_resource_tile(x):
      return x < len(resource_tiles)

    def is_citytiles(x):
      return len(resource_tiles) <= x < len(resource_tiles) + len(citytiles)

    def is_near_res_tiles(x):
      low = len(resource_tiles) + len(citytiles)
      high = low + len(near_resource_tiles)
      return low <= x < high

    def get_resource_weight(worker, resource_tile, dist, unit_night_count):
      # if worker.get_cargo_space_left() == 0:
        # return 0

      # Use dist - 1, because then the worker will get resource.
      target_night_count = get_night_count_by_dist(g.turn, dist-1, worker.cooldown,
                                                   get_unit_action_cooldown(worker))
      # TODO: add some buffer for safe arrival
      if unit_night_count < target_night_count:
        return -1

      wt = get_cell_resource_value(resource_tile, player)
      for newpos in get_neighbour_positions(resource_tile.pos, self.game.map):
        nb_cell = self.game.map.get_cell_by_pos(newpos)
        wt += get_cell_resource_value(nb_cell, player)

      research_done = int(is_cell_done_research(resource_tile, g.player))
      return (wt * research_done) / dist_decay(dist, g.game_map)


    def get_city_tile_weight(worker, citytile, dist, unit_night_count):
      FULL_WORKER_WEIGHT = 50
      CITYTILE_LOST_WEIGHT = 10

      target_night_count = get_night_count_by_dist(g.turn, dist, worker.cooldown,
                                                   get_unit_action_cooldown(worker))
      if unit_night_count < target_night_count:
        return -10

      # TODO: consider change strategy after research point reach coal or uranium.
      # woker won't step on worker set when has resource
      # if self.is_wood_citytile(citytile) and worker_total_cargo(worker) > 0:
        # wt = 0
      # else:
      wt = 0.1

      city = g.player.cities[citytile.cityid]
      city_will_last = not city_wont_last_at_nights(g.turn, city)
      round_night_count = get_night_count_this_round(g.turn)
      if is_night(g.turn) and city_will_last and unit_night_count < round_night_count:
        wt += 1

      if worker.get_cargo_space_left() == 0:
        # wt += FULL_WORKER_WEIGHT
        city = g.player.cities[citytile.cityid]
        if city_wont_last_at_nights(g.turn, city):
          wt += len(city.citytiles) * CITYTILE_LOST_WEIGHT
      return wt / dist_decay(dist, g.game_map)

    def get_near_resource_tile_weight(worker, near_resource_tile, dist,
                                      unit_night_count):
      target_night_count = get_night_count_by_dist(g.turn, dist, worker.cooldown,
                                                   get_unit_action_cooldown(worker))
      # TODO: add some buffer for safe arri0
      if unit_night_count < target_night_count:
        return -1

      res = 0
      for pos in get_neighbour_positions(near_resource_tile.pos, g.game_map):
        nb_cell = g.game_map.get_cell_by_pos(pos)
        if nb_cell.has_resource():
          res += get_cell_resource_value(nb_cell, player)

      build_city_bonus = False
      t = self.game.turn % CIRCLE_LENGH
      days_left = BUILD_CITYTILE_ROUND - t - worker.cooldown
      if (worker_enough_cargo_to_build(worker, res)
          and t < BUILD_CITYTILE_ROUND
          and days_left >= (dist - 1) * get_unit_action_cooldown(worker) + 1):
        build_city_bonus = True

      v = 0.1 + res

      cargo_full_rate = worker_total_cargo(worker) / WORKER_RESOURCE_CAPACITY
      v = v * (np.e ** cargo_full_rate)
      if build_city_bonus:
        v = v + 10000

      # Build city as fast as possible.
      return v / (dist + 1)
      # return v / dist_decay(dist, g.map)

    def get_worker_tile_weight(worker, target):
      if worker.pos == target.pos:
        return 0.1
      return -1

    # Value matrix for ship target assginment
    # * row: workers
    # * column: point of interests
    weights = np.zeros((len(workers), len(targets)))

    # MAIN_PRINT
    print((f'*turn={g.turn}, #worker={len(workers)}, #citytile={n_citytile} '
           f'research={g.player.research_points}'), file=sys.stderr)
    for i, worker in enumerate(workers):
      unit_night_count = cargo_night_endurance(worker.cargo, get_unit_upkeep(worker))
      for j, target in enumerate(targets):
        dist = self.shortet_paths[worker.id].shortest_dist(target.pos)
        if dist >= MAX_PATH_WEIGHT:
          continue

        v = 0
        if is_resource_tile(j):
          v = get_resource_weight(worker, target, dist, unit_night_count)
        elif is_citytiles(j):
          v = get_city_tile_weight(worker, target, dist, unit_night_count)
        elif is_near_res_tiles(j):
          # near resoure tiles
          v = get_near_resource_tile_weight(worker, target, dist,
                                            unit_night_count)
        else:
          v = get_worker_tile_weight(worker, target)
          # if v > 0:
            # print(f'nct_wt[{target.pos}]={v}', file=sys.stderr)

        # if DEBUG:
          # t = annotate.text(target.pos.x, target.pos.y, f'{v:.1f}')
          # print(t, file=sys.stderr)
          # self.actions.append(t)

        weights[i, j] = v


    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    for worker_idx, target_idx in zip(rows, cols):
      worker = workers[worker_idx]
      target = targets[target_idx]
      worker.target_pos = target.pos

      if DEBUG:
        # print(f'worker[{worker_idx}], v={weights[worker_idx, target_idx]}', file=sys.stderr)
        x = annotate.x(worker.pos.x, worker.pos.y)
        c = annotate.circle(target.pos.x, target.pos.y)
        line = annotate.line(worker.pos.x, worker.pos.y, target.pos.x, target.pos.y)
        self.actions.extend([x, c, line])



  def compute_worker_moves(self):
    g = self.game
    player = g.player
    workers = [unit for unit in player.units
               if unit.is_worker() and not unit.has_planned_action]

    # unit_target_positions = set()
    # for w in workers:
      # if w.target_pos:
        # unit_target_positions.add(w.target_pos)

    def compute_weight(worker, next_position, shortest_path,
                       shortest_path_points):
      if worker.target_pos is None or shortest_path_points is None:
        return 0

      v = 0
      next_cell = g.game_map.get_cell_by_pos(next_position)
      # try not step on citytile
      # if (next_cell.citytile is not None
          # and next_cell.citytile.team == g.player.team
          # and is_night(g.turn)):
        # citytile = next_cell.citytile
        # city = g.player.cities[citytile.cityid]
        # if city_wont_last_at_nights(g.turn, city):
          # v -= 5

      if next_position in shortest_path_points:
        v += 1

        target_dist = shortest_path.shortest_dist(worker.target_pos)
        worker_to_next_pos_dist = shortest_path_points[next_position]
        next_pos_to_target_dist = target_dist - worker_to_next_pos_dist
        if next_pos_to_target_dist < target_dist:
          v += 10


        # try step on resource
        rv = get_cell_resource_value(next_cell, g.player)
        v += int(rv > 0)

        # demote target cell one next move

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
    C = np.ones((len(workers), len(next_positions))) * -1
    for worker_idx, worker in enumerate(workers):
      shortest_path = self.shortet_paths[worker.id]

      shortest_path_points = None
      if worker.target_pos is not None:
        shortest_path_points = shortest_path.compute_shortest_path_points(worker.target_pos)
        # if DEBUG:
          # print(f'  w[{worker.id}]={worker.pos}, target={worker.target_pos}, S_path={shortest_path_points}', file=sys.stderr)

      for next_position in gen_next_positions(worker):
        wt = compute_weight(worker, next_position,
                            shortest_path, shortest_path_points)
        C[worker_idx, position_to_indices[next_position]] = wt
        # if DEBUG:
          # print(f'w[{worker.id}]  goto {next_position} wt = {wt}', file=sys.stderr)

    # print(f'turn={g.turn}, compute_worker_moves before linear_sum_assignment',
          # file=sys.stderr)
    rows, cols = scipy.optimize.linear_sum_assignment(C, maximize=True)
    for worker_idx, poi_idx in zip(rows, cols):
      worker = workers[worker_idx]
      if not worker.can_act():
        continue

      next_position = next_positions[poi_idx]
      move_dir = worker.pos.direction_to(next_position)
      self.add_unit_action(worker, worker.move(move_dir))


  def try_build_citytile(self):
    t = self.game.turn % CIRCLE_LENGH
    for unit in self.game.player.units:
      if not unit.is_worker():
        continue

      # Sitting on th cell of target position for city building.
      if (unit.can_act() and unit.can_build(self.game.game_map)
          and unit.target_pos and unit.target_pos == unit.pos):
        cell = self.game_map.get_cell_by_pos(unit.pos)
        if cell.is_near_resource:
          self.add_unit_action(unit, unit.build_city())

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
      if total_unit_count < total_tile_count:
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
      # forbidden_positions = (self.citytile_positions
                             # if (worker_total_cargo(unit) > 0
                                 # and is_night(self.game.turn))
                             # else empty_set)
      shortest_path = ShortestPath(self.game, unit.pos, empty_set)
      shortest_path.compute()
      self.shortet_paths[unit.id] = shortest_path

  def execute(self):
    actions = self.actions
    g = self.game
    player = g.player

    self.update_wood_citytile_positions(g.player_city_tiles)
    self.compute_unit_shortest_paths()
    # print(f'turn={g.turn}, compute_shortest_path done', file=sys.stderr)

    self.compute_citytile_actions()
    # print(f'turn={g.turn}, compute_citytile_actions done', file=sys.stderr)
    self.assign_worker_target()
    # print(f'turn={g.turn}, assign_worker_target done', file=sys.stderr)

    self.try_build_citytile()
    # print(f'turn={g.turn}, try_build_citytile done', file=sys.stderr)
    self.compute_worker_moves()
    # print(f'turn={g.turn}, compute_worker_moves done', file=sys.stderr)


_strategy = Strategy()

def agent(observation, configuration):
  _strategy.update(observation, configuration)
  _strategy.execute()
  return _strategy.actions
