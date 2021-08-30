"""
* Add units (enemy and mine) to cell and check blocking
"""

import math, sys
from collections import defaultdict, deque

import numpy as np
import scipy.optimize

from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

DEBUG = True

BUILD_CITYTILE_ROUND = 20


MAX_PATH_WEIGHT = 99999

def params(key):
  return GAME_CONSTANTS['PARAMETERS'][key]


DIRECTIONS = Constants.DIRECTIONS
DAY_LENGTH = params('DAY_LENGTH')
NIGHT_LENGTH = params('NIGHT_LENGTH')
CIRCLE_LENGH = DAY_LENGTH + NIGHT_LENGTH

UNIT_ACTION_COOLDOWN = params('UNIT_ACTION_COOLDOWN')
WORKER_ACTION_COOLDOWN = UNIT_ACTION_COOLDOWN["WORKER"]
CITY_BUILD_COST = params('CITY_BUILD_COST')


def get_resource_to_fuel_rate(resource):
  return params('RESOURCE_TO_FUEL_RATE')[resource.type.upper()]


def get_unit_action_cooldown(unit):
  ut = 'WORKER' if unit.type == 0 else 'CART'
  return UNIT_ACTION_COOLDOWN[ut]


def is_within_map_range(pos, map):
  return 0 <= pos.x < map.width and 0 <= pos.y < map.height

def worker_total_cargo(worker):
  cargo = worker.cargo
  return (cargo.wood + cargo.coal + cargo.uranium)


def worker_enough_cargo_to_build(worker):
  return worker_total_cargo(worker) >= CITY_BUILD_COST

def get_neighbour_positions(pos, map):
  check_dirs = [
    DIRECTIONS.NORTH,
    DIRECTIONS.EAST,
    DIRECTIONS.SOUTH,
    DIRECTIONS.WEST,
  ]
  positions = []
  for direction in check_dirs:
    newpos = pos.translate(direction, 1)
    if not is_within_map_range(newpos, map):
      continue
    positions.append(newpos)
  return positions


def cell_has_opponent_citytile(cell, game):
  citytile = cell.citytile
  return citytile is not None and citytile.team == game.opponent_id


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

  def __init__(self, game, start_pos):
    self.game = game
    self.map = game.map
    self.start_pos = start_pos
    self.dist = np.ones((self.map.width, self.map.height)) * MAX_PATH_WEIGHT

  def compute(self):
    q = deque([self.start_pos])
    self.dist[self.start_pos.x, self.start_pos.y] = 0
    # return

    total_append = 0
    while q:
      cur = q.popleft()
      cur_dist = self.dist[cur.x, cur.y]
      for newpos in get_neighbour_positions(cur, self.map):
        nb_cell = self.map.get_cell_by_pos(newpos)

        # TODO: also check enemy and my unit in cooldown
        if cell_has_opponent_citytile(nb_cell, self.game):
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
      for newpos in get_neighbour_positions(cur, self.map):
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

  @property
  def map(self):
    return self.game.map

  def update(self, observation, configuration):
    self.game.update(observation, configuration)

    # Clear up actions for current step.
    self.actions = []

    for unit in self.game.player.units:
      unit.next_action = None
      unit.target_pos = None


  def cell_near_resource(self, cell):
    for pos in get_neighbour_positions(cell.pos, self.map):
      nb_cell = self.map.get_cell_by_pos(pos)
      if nb_cell.has_resource():
        return True
    return False

  def assign_worker_target(self):
    MAX_UNIT_PER_CITY = 3
    g = self.game
    player = g.player

    workers = [unit for unit in player.units if unit.is_worker()]

    # TODO: remove unit occupied by opponent_unit and opponent_citytile
    resource_tiles: list[Cell] = []
    near_resource_tiles = []
    for y in range(g.map_height):
        for x in range(g.map_width):
            cell = g.map.get_cell(x, y)
            if cell.has_resource():
              resource_tiles.append(cell)
            if (not cell.has_resource()
                and cell.citytile is None
                and self.cell_near_resource(cell)):
              near_resource_tiles.append(cell)


    city_tiles = g.player_city_tiles * MAX_UNIT_PER_CITY

    targets = resource_tiles + city_tiles + near_resource_tiles

    def is_resource_tile(x):
      return x < len(resource_tiles)

    def is_citytiles(x):
      return len(resource_tiles) <= x < len(resource_tiles) + len(city_tiles)

    def get_cell_resource_value(cell):
      if not cell.has_resource():
        return 0

      resource = cell.resource
      if (resource.type == Constants.RESOURCE_TYPES.COAL
          and not player.researched_coal()):
        return 0
      if (resource.type == Constants.RESOURCE_TYPES.URANIUM
          and not player.researched_uranium()):
        return 0

      value = resource.amount * get_resource_to_fuel_rate(resource)
      return value

    def get_resource_weight(worker, resource_tile, dist):
      # TODO: do not go outside with empty resoure in the night
      if worker.get_cargo_space_left() == 0:
        return 0

      wt = get_cell_resource_value(resource_tile)
      for newpos in get_neighbour_positions(resource_tile.pos, self.game.map):
        nb_cell = self.game.map.get_cell_by_pos(newpos)
        wt += get_cell_resource_value(nb_cell)
      return wt / (dist + 1)

    def get_city_tile_weight(worker, citytile, dist):
      if worker.get_cargo_space_left() == 0:
        return 50000 / (dist + 1)
      return 0

    def get_near_resource_tile_weight(worker, near_resource_tile, dist):
      if not worker_enough_cargo_to_build(worker):
        return 0

      t = self.game.turn % CIRCLE_LENGH
      if t >= BUILD_CITYTILE_ROUND:
        return 0

      # won't arrive
      round_left = BUILD_CITYTILE_ROUND - t - worker.cooldown
      if round_left / get_unit_action_cooldown(worker) < dist:
        return 0

      res_wt = 0
      citytile_wt = 0
      for pos in get_neighbour_positions(near_resource_tile.pos, self.map):
        nb_cell = self.map.get_cell_by_pos(pos)
        if nb_cell.has_resource():
          res_wt += get_cell_resource_value(nb_cell)

        ct = nb_cell.citytile
        if ct and ct.team == self.game.player.team:
          citytile_wt += 1

      v = (res_wt + citytile_wt * 50) * 100
      # print(f' >> near_v={v}', file=sys.stderr)
      return v / (dist + 1)

    # Value matrix for ship target assginment
    # * row: workers
    # * column: point of interests
    weights = np.zeros((len(workers), len(targets)))

    print(f'turn={g.turn}, #worker={len(workers)},  #near_resource_tile_cnt={len(near_resource_tiles)}', file=sys.stderr)
    for i, worker in enumerate(workers):
      for j, target in enumerate(targets):
        dist = self.shortet_paths[worker.id].shortest_dist(target.pos)

        v = 0
        if is_resource_tile(j):
          v = get_resource_weight(worker, target, dist)
        elif is_citytiles(j):
          v = get_city_tile_weight(worker, target, dist)
        else:
          # near resoure tiles
          v = get_near_resource_tile_weight(worker, target, dist)
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
        a = annotate.circle(target.pos.x, target.pos.y)
        self.actions.append(a)

        a = annotate.line(worker.pos.x, worker.pos.y, target.pos.x, target.pos.y)
        self.actions.append(a)


  def compute_worker_moves(self):
    g = self.game
    player = g.player
    workers = [unit for unit in player.units
               if unit.is_worker() and unit.can_act()]

    def compute_weight(worker, next_position, shortest_path,
                       shortest_path_points):
      if worker.target_pos is None or shortest_path_points is None:
        return 0

      if next_position in shortest_path_points:
        v = 1

        target_dist = shortest_path.shortest_dist(worker.target_pos)
        worker_to_next_pos_dist = shortest_path_points[next_position]
        next_pos_to_target_dist = target_dist - worker_to_next_pos_dist
        if next_pos_to_target_dist < target_dist:
          v += 10
        return v
      return 0

    def gen_next_positions(worker):
      assert worker.can_act()
      # TODO: skip non-reachable positions?
      return [worker.pos] + get_neighbour_positions(worker.pos, g.map)

    next_positions = {
      pos
      for worker in workers
      for pos in gen_next_positions(worker)
    }

    def duplicate_positions(positions):
      for pos in positions:
        cell = self.game.map.get_cell_by_pos(pos)
        if cell.citytile is not None and cell.citytile.team == self.game.id:
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

    position_to_index = get_position_to_index()
    C = np.zeros((len(workers), len(next_positions)))
    for worker_idx, worker in enumerate(workers):
      shortest_path = self.shortet_paths[worker.id]

      shortest_path_points = None
      if worker.target_pos is not None:
        shortest_path_points = shortest_path.compute_shortest_path_points(worker.target_pos)
        # if DEBUG:
          # print('  ' , worker_idx, shortest_path_points, file=sys.stderr)

      for next_position in gen_next_positions(worker):
        for poi_idx in position_to_index[next_position]:
          poi_idx = position_to_index[next_position]
          wt = compute_weight(worker, next_position,
                              shortest_path, shortest_path_points)
          C[worker_idx, poi_idx] = wt

          # if DEBUG:
            # print(f'w[{worker_idx}]  goto {next_position} wt = {wt}', file=sys.stderr)
 
    # print(f'turn={g.turn}, compute_worker_moves before linear_sum_assignment',
          # file=sys.stderr)
    rows, cols = scipy.optimize.linear_sum_assignment(C, maximize=True)
    for worker_idx, poi_idx in zip(rows, cols):
      worker = workers[worker_idx]
      next_position = next_positions[poi_idx]

      move_dir = worker.pos.direction_to(next_position)
      self.actions.append(worker.move(move_dir))

    # print(f'turn={g.turn}, compute_worker_moves after linear_sum_assignment',
          # file=sys.stderr)


  def try_build_citytile(self):
    t = self.game.turn % CIRCLE_LENGH
    for unit in self.game.player.units:
      if not unit.is_worker():
        continue

      if unit.can_act() and unit.can_build(self.game.map) and t < BUILD_CITYTILE_ROUND:
        unit.target_pos = None
        self.actions.append(unit.build_city())

  def compute_citytile_actions(self):
    player = self.game.player
    total_tile_count = player.city_tile_count
    total_unit_count = len(player.units)

    for _, city in player.cities.items():
      for citytile in city.citytiles:
        if citytile.can_act():
          if total_unit_count < total_tile_count:
            self.actions.append(citytile.build_worker())
            total_unit_count += 1
          else:
            self.actions.append(citytile.research())

  def compute_unit_shortest_paths(self):
    self.shortet_paths = {}

    for unit in self.game.player.units:
      shortest_path = ShortestPath(self.game, unit.pos)
      shortest_path.compute()
      self.shortet_paths[unit.id] = shortest_path


  def execute(self):
    actions = self.actions
    g = self.game
    player = g.player

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
