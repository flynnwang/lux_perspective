"""

* Limit maximum amount when compute resource weight
* Bugfix on worker_enough_cargo_to_build() from fuel to amount
* Support Dying woker weights on global and local moves.
* Check near_resource_tile target blocked by citytile.
* Only pass through enemy unit which can't act (but not player unit)
* Try lower city tile boost value for full cargo worker: 1
* Raise default weight 0.1 for any resource tile

Total Matches: 443 | Matches Queued: 11
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
/Users/flynnwang/dev/playground/versions/fast_city_far_resource/main.py | W4frPn0iVPSo   | 25.2599588      | μ=27.776, σ=0.839  | 443
/Users/flynnwang/dev/playground/lux_perspective/main.py | lmojnyTBNFB9   | 19.7072696      | μ=22.224, σ=0.839  | 443


* Fix resource.amount
Total Matches: 107 | Matches Queued: 11
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
/Users/flynnwang/dev/playground/versions/fast_city_far_resource/main.py | VitC7Rawf5Ee   | 24.4715737      | μ=27.310, σ=0.946  | 107
/Users/flynnwang/dev/playground/lux_perspective/main.py | rSHgEc2FvHXq   | 19.8509722      | μ=22.690, σ=0.946  | 107


* Only build city tile before 24
* Move to citytile to save city giving 20+ boost
* use small decay for resource (encourge explore)
* Try use total resource weight (as it drops, agent will explore)


# Doing
* Raise priority of near_resource_tile targeted worker





# TODO
* give some weight to not finished research resource (arrive at them earlier)
* Add resource weight to city tile.


TODO
* Limit max worker to build, but not limit citytile


TODO:
* Add units (enemy and mine) to cell and check blocking
# TODO: build city away from near resource tile
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

BUILD_CITYTILE_ROUND = 24

MAX_PATH_WEIGHT = 99999

# UES_RESOURCE_TOTAL_FUEL = True


def dist_decay(dist, game_map):
  decay = game_map.width
  # decay = 24
  return (dist / decay) + 1


def worker_enough_cargo_to_build(worker, collected_amount):
  return (worker_total_cargo(worker) + collected_amount >= CITY_BUILD_COST)


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


  # amount: how many the worker can pick up
  amount = get_worker_collection_rate(resource)
  amount = min([amount, resource.amount])
  if unit:
    amount = min(amount, unit.get_cargo_space_left())

  # Fuel: total fuel of this cell
  fuel = resource.amount * get_resource_to_fuel_rate(resource)
  return amount, fuel

  # Local weight
  # amount = get_worker_collection_rate(resource)
  # amount = min([amount, resource.amount])
  # if unit:
    # amount = min(amount, unit.get_cargo_space_left())
  # fuel = amount * get_resource_to_fuel_rate(resource)
  # return amount, fuel



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


# TODO: maybe rename?
def get_unit_collection_values(cell, player, unit, game_map):
  amount, fuel = get_cell_resource_values(cell, player, unit)
  for newpos in get_neighbour_positions(cell.pos, game_map):
    nb_cell = game_map.get_cell_by_pos(newpos)
    a, f = get_cell_resource_values(nb_cell, player, unit)
    amount += a
    fuel += f
  return amount, fuel


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
        if (hasattr(nb_cell, 'unit') and
            nb_cell.unit.team == self.game.opponent_id and not nb_cell.unit.can_act()):
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


class Strategy:

  def __init__(self):
    self.actions = []
    self.game = LuxGame()
    self.citytile_positions = set()

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
          # TODO: move it to a common place
          cell.citytile.cell = cell
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

    # mark_boundary_resource(resource_tiles, g.game_map)

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

    MAX_WEIGHT_VALUE = 50000
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
        wt += MAX_WEIGHT_VALUE * 2  # TODO: why x 2?

      _, fuel = get_unit_collection_values(resource_tile, player, worker,
                                           g.game_map)

      wt += fuel


      # add weight to empty worke
      cargo_full_rate = (1 - worker_cargo_full_rate(worker))
      boost = (np.e ** cargo_full_rate)
      wt *= boost

      return wt / dist_decay(dist, g.game_map)

    def get_city_tile_weight(worker, citytile, dist, unit_night_count):
      """
      1. collect fuel
      2. protect dying worker [at night]
      """
      CITYTILE_LOST_WEIGHT = 1000

      # TODO: It's asuming dist are full of danger, but it could be move inside citytile.
      target_night_count = get_night_count_by_dist(g.turn, dist, worker.cooldown,
                                                   get_unit_action_cooldown(worker))
      if unit_night_count < target_night_count:
        return -1

      # amount, fuel = get_unit_collection_values(citytile.cell, player, worker,
                                                # g.game_map)
      # wt = fuel # base score.
      wt = 0

      # Try to hide in the city if worker will run out of fuel at night
      city = g.player.cities[citytile.cityid]
      city_will_last = not city_wont_last_at_nights(g.turn, city)
      if (self.circle_turn >= BUILD_CITYTILE_ROUND and worker.is_dying and city_will_last):
        wt += MAX_WEIGHT_VALUE / 10

      # if worker.id == 'u_12':
        # print(f"w[{worker.id}], t[{citytile.pos}], turn_c={self.circle_turn >= BUILD_CITYTILE_ROUND}, dying={worker.is_dying}, city_last={city_will_last}", file=sys.stderr)

      # TODO(wangfei): estimate worker will full
      boost_city_lost_wt = False
      if self.circle_turn < BUILD_CITYTILE_ROUND and worker.get_cargo_space_left() == 0:
        boost_city_lost_wt = True

      # TODO: support night
      # Save more citytile if worker has enough resource to save it
      # - enough time to arrive
      # - with substential improvement of its living
      unit_time_cost = (dist - 1) * get_unit_action_cooldown(worker) + 1
      days_left = DAY_LENGTH - self.circle_turn - worker.cooldown
      city_left_days = math.ceil(city.fuel / city.light_upkeep)
      woker_fuel = worker_total_fuel(worker)
      city_left_days_deposited = math.ceil((city.fuel + woker_fuel) / city.light_upkeep)
      if (days_left >= unit_time_cost
          and city_left_days < NIGHT_LENGTH
          and city_left_days_deposited > NIGHT_LENGTH
          and city_left_days_deposited - city_left_days >= 20):
        boost_city_lost_wt = True

      if boost_city_lost_wt:
        wt += 10000
        wt += CITYTILE_LOST_WEIGHT * len(city.citytiles)

      return wt / dist_decay(dist, g.game_map)
      # return wt / (dist + 1)

      # TODO(wangfei): merge near resource tile and resource tile weight functon
    def get_near_resource_tile_weight(worker, near_resource_tile, dist,
                                      unit_night_count):
      target_night_count = get_night_count_by_dist(g.turn, dist, worker.cooldown,
                                                   get_unit_action_cooldown(worker))
      # TODO: add some buffer for safe arri0
      if unit_night_count < target_night_count:
        return -1

      amount, fuel = get_unit_collection_values(near_resource_tile, player,
                                                worker, g.game_map)

      wt = fuel

      # TODO: maybe time consuming
      shortest_path = self.shortet_paths[worker.id]
      blocked_by_citytile = shortest_path.path_blocked_by_citytile(near_resource_tile.pos)
      if not blocked_by_citytile:
        build_city_bonus = False
        days_left = BUILD_CITYTILE_ROUND - self.circle_turn - worker.cooldown
        if (worker_enough_cargo_to_build(worker, amount)
            and self.circle_turn < BUILD_CITYTILE_ROUND
            and days_left >= (dist - 1) * get_unit_action_cooldown(worker) + 1):
          build_city_bonus = True

        cargo_full_rate = (1 - worker_cargo_full_rate(worker))
        boost = (np.e ** cargo_full_rate)
        # wt *= boost

        # Too large the build city bonus will cause worker divergence from its coal mining position
        if build_city_bonus:
          wt += 1000
          wt *= 100

        # p = near_resource_tile.pos
        # if worker.id == 'u_1' and p in [Position(7, 3), Position(7, 1)]:
          # print(f"w[{worker.id}] - {near_resource_tile.pos}, blocked_by_citytile={blocked_by_citytile}, boost={boost}, fuel={fuel}", file=sys.stderr)

      # Try to hide next to resource grid.
      if is_resource_tile_can_save_dying_worker(near_resource_tile, worker):
        wt += MAX_WEIGHT_VALUE

      # return wt / dist_decay(dist, g.map)
      return wt / (dist + 1)

    def get_worker_tile_weight(worker, target):
      if worker.pos == target.pos:
        return 0.1
      return -1

    # Value matrix for ship target assginment
    # * row: workers
    # * column: point of interests
    weights = np.zeros((len(workers), len(targets)))

    # MAIN_PRINT
    print((f'*turn={g.turn}, #W={len(workers)}, #C={n_citytile} '
           f'R={g.player.research_points}'),
          file=sys.stderr)
    for i, worker in enumerate(workers):
      unit_night_count = cargo_night_endurance(worker.cargo, get_unit_upkeep(worker))
      worker.unit_night_count = unit_night_count

      round_night_count = get_night_count_this_round(g.turn)
      worker.is_dying = unit_night_count < round_night_count

      # if worker.id == 'u_29':
        # print(f"w[{worker.id}], unit_night_count={unit_night_count}, round_night_count={round_night_count}", file=sys.stderr)

      for j, target in enumerate(targets):
        # print(f'----S, i={i}, j={j}, {worker.id}')
        shortest_path = self.shortet_paths[worker.id]
        dist = shortest_path.shortest_dist(target.pos)

        # if worker.id == 'u_29' and target.pos == Position(4, 7):
          # print(f'-- dist={dist} is_resource_tile={is_resource_tile(j)}, ct={is_citytiles(j)}, nrt={is_near_res_tiles(j)}')

        if dist >= MAX_PATH_WEIGHT:
          continue

        # if worker.id == 'u_29' and target.pos == Position(4, 7):
          # print(f' not skip ? dist={dist} MAX_PATH_WEIGHT={MAX_PATH_WEIGHT}, {dist >=MAX_PATH_WEIGHT}')

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
        # if worker.id == 'u_1' and v > 100:
          # print(f"w[{worker.id}], t[{target.pos}], wt={v:.1f}", file=sys.stderr)
        # if worker.id == 'u_29' and target.pos == Position(4, 6):
          # print(f"w[{worker.id}], t[{target.pos}], wt={v}", file=sys.stderr)
        # if worker.id == 'u_29' and target.pos == Position(0, 0):
          # print(f"w[{worker.id}], t[{target.pos}], wt={v}", file=sys.stderr)

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
      """Only workers next 5 positions will be computed here."""
      if worker.target_pos is None or shortest_path_points is None:
        return 0

      v = 0
      # Priority all positions of th dying worker, let others make room for him.
      if worker.is_dying:
        v += 100

      # Add priorixy to resource collection worker.
      target_cell = g.game_map.get_cell_by_pos(worker.target_pos)
      if target_cell.has_resource():
        v += 50

      next_cell = g.game_map.get_cell_by_pos(next_position)
      # try not step on citytile: not good
      # if (next_cell.citytile is not None
          # and next_cell.citytile.team == g.player.team
          # and is_night(g.turn)):
        # citytile = next_cell.citytile
        # city = g.player.cities[citytile.cityid]
        # if city_wont_last_at_nights(g.turn, city):
          # v -= 0.1

      fuel = 0
      if next_position in shortest_path_points:
        v += 1

        target_dist = shortest_path.shortest_dist(worker.target_pos)
        worker_to_next_pos_dist = shortest_path_points[next_position]
        next_pos_to_target_dist = target_dist - worker_to_next_pos_dist
        if next_pos_to_target_dist < target_dist:
          v += 10

        # Try step on resource: the worker version is better, maybe because
        # other worker can use that.
        # amount, fuel = get_cell_resource_values(next_cell, g.player, unit=None)
        amount, fuel = get_cell_resource_values(next_cell, g.player, unit=worker)
        if fuel > 0:
          v += 1

        # demote target cell one next move

      # if worker.id == 'u_1':
        # print(f"w[{worker.id}], next[{next_position}], v={v}, target={worker.target_pos}, fuel={fuel}", file=sys.stderr)

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
      if (self.circle_turn < BUILD_CITYTILE_ROUND
          and unit.can_act()
          and unit.can_build(self.game.game_map)
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
