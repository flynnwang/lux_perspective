import functools
import heapq
import math, sys
import time
from collections import defaultdict, deque
from copy import deepcopy

import numpy as np
import scipy.optimize

from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
from utility3 import *

DEBUG = True

DRAW_UNIT_ACTION = 1
DRAW_UNIT_CLUSTER_PAIR = 1

DRAW_UNIT_LIST = ['u_2']
MAP_POS_LIST = [(0, 13), (0, 12)]

MAP_POS_LIST = [Position(x, y) for x, y in MAP_POS_LIST]
DRAW_UNIT_TARGET_VALUE = 0
DRAW_UNIT_MOVE_VALUE = 0
DRAW_QUICK_PATH_VALUE = 0

# TODO: add more
BUILD_CITYTILE_ROUND = CIRCLE_LENGH

MAX_PATH_WEIGHT = 99999

MAX_UNIT_NUM = 71

MAX_UNIT_PER_CITY = 8

KEEP_RESOURCE_OPEN = 1

# def prt(line, file=sys.stderr):
# print(line, file=file)

prt = print


def timeit(func):

  def dec(*args, **kwargs):
    t1 = time.time()
    r = func(*args, **kwargs)
    t2 = time.time()
    if not DRAW_UNIT_LIST:
      prt(f"f[{func.__name__}], t={(t2-t1):.2f}", file=sys.stderr)
    return r

  return dec


@functools.lru_cache(maxsize=1024, typed=False)
def dd(dist, r=1.8):
  return r**dist


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


N9_DIRS = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]


def get_nb9_positions(pos, game_map, return_cell=True):
  positions = []
  for dx, dy in N9_DIRS:
    newpos = Position(pos.x + dx, pos.y + dy)
    if is_within_map_range(newpos, game_map):
      if return_cell:
        newpos = game_map.get_cell_by_pos(newpos)
      positions.append(newpos)
  return positions


def is_resource_researched(resource,
                           player,
                           move_days=0,
                           surviving_turns=0,
                           debug=False):
  if resource is None:
    return False
  plus = max(move_days, surviving_turns) * player.avg_research_point_growth
  if debug:
    prt(f' researched_coal={player.researched_coal(plus)}, '
        f'dR={player.avg_research_point_growth}, move_days={move_days},'
        f' surviving_turns={surviving_turns}, plus={plus}')
  if (resource.type == Constants.RESOURCE_TYPES.COAL and
      not player.researched_coal(plus)):
    return False
  if (resource.type == Constants.RESOURCE_TYPES.URANIUM and
      not player.researched_uranium(plus)):
    return False
  return True


def resource_researched_wait_turns(resource,
                                   player,
                                   move_days,
                                   surviving_turns,
                                   debug=False):
  if resource is None:
    return 0
  if resource.type == Constants.RESOURCE_TYPES.WOOD:
    return 0

  def waiting_turns(res_require_points):
    more_points = res_require_points - player.research_points
    more_points = max(0, more_points)
    point_growth_rate = player.avg_research_point_growth + 0.01
    wait_turns = more_points / point_growth_rate

    if debug:
      prt(f"move_days={move_days}, wait_turns={wait_turns}, surviving_turns={surviving_turns}"
         )
    if move_days + surviving_turns < wait_turns:
      # if wait_turns > surviving_turns:
      return -1
    return wait_turns

  if resource.type == Constants.RESOURCE_TYPES.COAL:
    return waiting_turns(COAL_RESEARCH_POINTS)
  if resource.type == Constants.RESOURCE_TYPES.URANIUM:
    return waiting_turns(URANIUM_RESEARCH_POINTS)

  assert False, f"resource type not found: {resource.type}"
  return 0


MAX_WAIT_RESORUCE_TURNS = CIRCLE_LENGH


@functools.lru_cache(maxsize=4096)
def get_cell_resource_values(cell,
                             player,
                             unit=None,
                             move_days=0,
                             surviving_turns=0,
                             debug=False):
  # Returns: (amount, fuel_weight)
  if not cell.has_resource():
    return 0, 0

  resource = cell.resource
  wait_turns = resource_researched_wait_turns(resource,
                                              player,
                                              move_days,
                                              surviving_turns,
                                              debug=debug)
  if debug:
    prt(f'get_cell_resource_values: [{cell.pos}] wait_turns={wait_turns}')
  if wait_turns < 0:
    if debug:
      prt(f' return from wait_turns: {wait_turns}')
    return 0, 0
  if wait_turns > MAX_WAIT_RESORUCE_TURNS:
    if debug:
      prt(f' such a long wait: {wait_turns}')
    return 0, 0

  amount = get_worker_collection_rate(resource)
  amount = min(amount, resource.amount)
  if unit:
    amount = min(amount, unit.get_cargo_space_left())
  fuel = amount * get_resource_to_fuel_rate(resource)
  return amount, fuel / dd(move_days + wait_turns)


# TODO(wangfei): try more accurate estimate
def resource_cell_added_surviving_nights(cell, upkeep, game):
  turns = resource_surviving_nights(game.turn, cell.resource, upkeep)
  for nb_cell in get_neighbour_positions(cell.pos,
                                         game.game_map,
                                         return_cell=True):
    turns += resource_surviving_nights(game.turn, nb_cell.resource, upkeep)
  return turns


@functools.lru_cache(maxsize=4096)
def get_one_step_collection_values(cell,
                                   player,
                                   game,
                                   move_days=0,
                                   surviving_turns=0,
                                   unit=None,
                                   debug=False):
  game_map = game.map
  amount, fuel_wt = get_cell_resource_values(cell,
                                             player,
                                             move_days=move_days,
                                             surviving_turns=surviving_turns,
                                             unit=unit,
                                             debug=debug)
  # if debug:
  # prt(f" main cell value [{cell.pos}]: amt={amount} fuel={fuel_wt}")

  # Use neighbour average as resource weight
  nb_fuel_wt = 0
  nb_count = 0
  nb_amt = 0
  for nb_cell in get_neighbour_positions(cell.pos, game_map, return_cell=True):
    a, f = get_cell_resource_values(nb_cell,
                                    player,
                                    move_days=move_days,
                                    surviving_turns=surviving_turns,
                                    unit=unit,
                                    debug=debug)
    # TODO: why max?
    nb_amt = max(nb_amt, a)
    nb_fuel_wt += f
    if a > 0:
      nb_count += 1

  # if nb_count > 0:
  # fuel_wt += nb_fuel_wt / nb_count

  # if debug:
  # prt(f" nb value [{cell.pos}]: amt={nb_amt} fuel={nb_fuel_wt}")
  return amount + nb_amt, fuel_wt + nb_fuel_wt


def collect_amount_at_cell(cell, player, game_map):
  amount, _ = get_cell_resource_values(cell, player)
  for nb_cell in get_neighbour_positions(cell.pos, game_map, return_cell=True):
    nb_amt, _ = get_cell_resource_values(nb_cell, player)
    amount += nb_amt
  return amount


def get_player_city_by_citytile(citytile, game):
  return game.player.cities[citytile.cityid]


@functools.lru_cache(maxsize=1024)
def cell_has_buildable_neighbour(cell, game_map):
  for nb_cell in get_neighbour_positions(cell.pos, game_map, return_cell=True):
    if (not nb_cell.has_resource() and nb_cell.citytile is None):
      return True
  return False


@functools.lru_cache(maxsize=1024)
def collect_target_cells(turn, game):
  target_cells = []

  game_map = game.game_map
  for y in range(game.map_height):
    for x in range(game.map_width):
      cell = game_map.get_cell(x, y)

      is_target_cell = False
      if cell_has_player_citytile(cell, game):
        cell.citytile.cell = cell
        citytiles = [deepcopy(cell)
                    ] + [deepcopy(cell)] * (MAX_UNIT_PER_CITY - 1)
        citytiles[0].is_first_citytile = True  # only boost for first tile.
        target_cells.extend(citytiles)
      elif cell.has_resource():
        is_target_cell = True
      elif cell.is_near_resource:
        is_target_cell = True
      elif cell.n_citytile_neighbour > 0:
        is_target_cell = True
      elif (cell.unit and cell.unit.team == game.player.team):
        is_target_cell = True

      if is_target_cell:
        cell.has_buildable_neighbour = cell_has_buildable_neighbour(
            cell, game_map)
        target_cells.append(cell)
  return target_cells


# After arrival, resource is enough to reach the next circle.
@functools.lru_cache(maxsize=4096)
def estimate_resource_night_count(worker,
                                  cell,
                                  upkeep,
                                  arrival_turns,
                                  surviving_turns,
                                  debug=False):
  if not cell.has_resource():
    return 0

  # quick_path, dest_turns = _strategy.select_quicker_path(worker, cell.pos)
  # if dest_turns >= MAX_PATH_WEIGHT:
  # return 0
  # surviving_turns = get_surviving_turns_at_cell(worker, quick_path, cell, debug=debug)

  wait_turns = resource_researched_wait_turns(
      cell.resource,
      _strategy.player,  # dirty.
      arrival_turns,
      surviving_turns=surviving_turns)
  if debug:
    print(
        f' {worker.id} arrival_turns={arrival_turns}, survive_turns={surviving_turns} at cell[{cell.pos}], wait_turns={wait_turns}]'
    )
  if wait_turns < 0:  # or wait_turns > arr:
    return 0
  cargo = resource_to_cargo(cell.resource)
  return cargo_night_endurance(cargo, upkeep)


@functools.lru_cache(maxsize=4096)
def estimate_cell_night_count(worker,
                              cell,
                              game_map,
                              arrival_turns,
                              surviving_turns,
                              debug=False):
  upkeep = get_unit_upkeep(worker)
  nights = estimate_resource_night_count(worker,
                                         cell,
                                         upkeep,
                                         arrival_turns,
                                         surviving_turns,
                                         debug=debug)
  for nb_cell in get_neighbour_positions(cell.pos, game_map, return_cell=True):
    nights += estimate_resource_night_count(worker,
                                            nb_cell,
                                            upkeep,
                                            arrival_turns,
                                            surviving_turns,
                                            debug=debug)
  return nights


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
    return [
        citytile for _, city in self.player.cities.items()
        for citytile in city.citytiles
    ]

  @property
  def is_night(self):
    return not is_day(self.turn)

  @property
  def is_day(self):
    return is_day(self.turn)

  @property
  def circle_turn(self):
    return self.turn % CIRCLE_LENGH

  @property
  def days_this_round(self):
    return get_day_count_this_round(self.turn)

  @property
  def night_in_round(self):
    return get_night_count_this_round(self.turn)

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

  MAX_SEARCH_DIST = 12

  def __init__(self, game, unit, ignore_unit=False):
    self.game = game
    self.game_map = game.game_map
    self.unit = unit
    self.start_pos = self.unit.pos
    self.dist = np.ones(
        (self.game_map.width, self.game_map.height)) * MAX_PATH_WEIGHT
    self.ignore_unit = ignore_unit

  @property
  def player(self):
    return self.game.players[self.unit.team]

  @property
  def opponent(self):
    return self.game.players[1 - self.unit.team]

  def compute(self):
    q = deque([self.start_pos])
    self.dist[self.start_pos.x, self.start_pos.y] = 0

    while q:
      cur = q.popleft()
      cur_dist = self.dist[cur.x, cur.y]
      for nb_cell in get_neighbour_positions(cur,
                                             self.game_map,
                                             return_cell=True):
        # Can not go pass through enemy citytile.
        if cell_has_target_player_citytile(nb_cell, self.opponent):
          continue

        # Skip opponent unit.
        if (not self.ignore_unit and nb_cell.unit and
            nb_cell.unit.team == self.opponent.team):
          continue

        newpos = nb_cell.pos
        nb_dist = self.dist[newpos.x, newpos.y]
        if cur_dist + 1 >= nb_dist:
          continue

        self.dist[newpos.x, newpos.y] = cur_dist + 1
        if cur_dist + 1 > self.MAX_SEARCH_DIST:
          continue

        q.append(newpos)
        # prt(f' start_from {self.start_pos}, append {newpos}, cur_dist={cur_dist}', file=sys.stderr)

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
      for nb_cell in get_neighbour_positions(cur,
                                             self.game_map,
                                             return_cell=True):
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
      for nb_cell in get_neighbour_positions(cur,
                                             self.game_map,
                                             return_cell=True):
        newpos = nb_cell.pos
        nb_dist = self.dist[newpos.x, newpos.y]
        if nb_dist == cur_dist - 1 and newpos not in path_positions:
          total_append += 1
          q.append(newpos)
          path_positions[newpos] = cur_dist - 1

    # prt(f'path points {self.start_pos}, totol_append={total_append}', file=sys.stderr)
    return path_positions


class SearchState:

  def __init__(self, turn, cell, cargo, cooldown):
    self.turn = turn
    self.cell = cell
    self.cargo = cargo
    self.cooldown = cooldown
    self.prev_positions = []
    self.deleted = False
    self.arrival_fuel = 0
    self.extra_wait_days = 0
    self.fuel = cargo_total_fuel(cargo)

  def __str__(self):
    return f'SearchState<t={self.turn}, [{self.cell.pos}] cd={self.cooldown}, arr_fuel={self.arrival_fuel}, cargo={str(self.cargo)}>'

  @property
  def pos(self):
    return self.cell.pos

  def get_surviving_turns(self, upkeep):
    last_nights = cargo_night_endurance(self.cargo, upkeep)
    return nights_to_last_turns(self.turn, last_nights)

  def __lt__(self, other):
    return self.turn < other.turn or (self.turn == other.turn and
                                      self.fuel > other.fuel)

  def __eq__(self, other):
    return self.turn == other.turn and self.fuel == other.fuel


@functools.lru_cache(maxsize=1024)
def get_res_type_to_resource(cell, game_map, player):
  # count the number of neighbour resource types (assume cell not change)
  collect_cells = (
      get_neighbour_positions(cell.pos, game_map, return_cell=True) + [cell])
  res_type_to_cells = defaultdict(list)
  for c in collect_cells:
    # TODO: Should i use estimate?
    if c.resource and is_resource_researched(c.resource, player):
      res_type_to_cells[c.resource.type.upper()].append(c.resource.amount)
  return res_type_to_cells


@functools.lru_cache(maxsize=1024)
def consume_cargo_resource(resource_amt, request_amt, fuel_rate):
  consumed_res_amt = min(int(math.ceil(request_amt / fuel_rate)), resource_amt)
  return (resource_amt - consumed_res_amt,
          max(request_amt - consumed_res_amt * fuel_rate, 0))


@functools.lru_cache(maxsize=1024)
def consume_worker_resource(wood0, coal0, uranium0, unit_upkeep):
  request_amt = unit_upkeep
  wood, request_amt = consume_cargo_resource(wood0, request_amt, WOOD_FUEL_RATE)
  coal, request_amt = consume_cargo_resource(coal0, request_amt, COAL_FUEL_RATE)
  uranium, request_amt = consume_cargo_resource(uranium0, request_amt,
                                                URANIUM_FUEL_RATE)
  if request_amt > 0:
    return None
  cargo = Cargo(wood, coal, uranium)
  return cargo


def sim_on_cell(turn,
                cargo,
                unit_type,
                next_cell,
                game,
                player,
                sim_turns,
                debug=False):
  """Sim one turn for worker move onto some cell, return cargo.

  1) collect resource
  2) drop resource to city.
  3) if night, make Units consume resources and CityTiles consume fuel
    - if unit no resource, removed
    - if city gone, unit gone with it
  """
  cargo = Cargo(cargo.wood, cargo.coal, cargo.uranium)
  capacity = get_unit_capacity_by_type(unit_type)

  def _request_amount(cargo_amount, res_type, n_res_type):
    left_amount = capacity - cargo_amount
    amt = min(int(math.ceil(left_amount / n_res_type)),
              WORKER_COLLECTION_RATE[res_type])
    return amt

  def collect_resource(cargo):
    cargo_amount = cargo_total_amount(cargo)
    if cargo_amount >= capacity:
      return

    if len(res_type_to_cells) == 0:
      return

    # TODO: may die because of other workers
    # For each type of resource, collect resource (assume no other wokers)
    for res_type in ALL_RESOURCE_TYPES:
      resource_amounts = res_type_to_cells.get(res_type)
      if not resource_amounts:
        continue

      req_amt = _request_amount(cargo_amount, res_type, len(resource_amounts))
      if req_amt == 0:
        continue

      collect_amt = sum(min(amt, req_amt) for amt in resource_amounts)
      add_resource_to_cargo(cargo, capacity, collect_amt, res_type)

  unit_upkeep = get_unit_upkeep_by_type(unit_type)

  def consume_worker_resource_by_cargo(cargo):
    return consume_worker_resource(cargo.wood, cargo.coal, cargo.uranium,
                                   unit_upkeep)

  res_type_to_cells = get_res_type_to_resource(next_cell, game.game_map, player)

  city_fuel = 0
  unit_on_citytile = cell_has_target_player_citytile(next_cell, player)
  if unit_on_citytile:
    city = player.cities[next_cell.citytile.cityid]
    nights = get_night_count_by_days(game.turn, turn - game.turn)
    city_fuel = city.fuel - nights * city.light_upkeep
    # TODO: check the meaning of run of of fuel
    if city_fuel < 0:
      unit_on_citytile = False

  assert unit_type == Constants.UNIT_TYPES.WORKER
  # assume city not change.
  for t in range(turn, turn + sim_turns):
    collect_resource(cargo)

    # 2) drop resource onto city.
    if unit_on_citytile:
      city_fuel += cargo_total_fuel(cargo)
      cargo.clear()

    if is_night(t):
      if not unit_on_citytile:
        new_cargo = consume_worker_resource_by_cargo(cargo)
        if new_cargo is None:
          return None
        cargo = new_cargo
      else:
        city_fuel -= city.light_upkeep
        if city_fuel < 0:
          return None
  return cargo


@functools.lru_cache(maxsize=4096, typed=False)
def get_surviving_turns_at_cell(worker, path, cell, debug=False):
  dest_state = path.get_dest_state(cell.pos)
  if debug:
    print(f" ## dest_state for {worker.id}@[{worker.pos}], s={str(dest_state)}")
  assert dest_state is not None
  return dest_state.get_surviving_turns(get_unit_upkeep(worker))


MAX_FUTURE_TURNS = CIRCLE_LENGH


class QuickestPath:
  """Find the path with (min_turns, max_cargo_resource)."""

  def __init__(self,
               game,
               worker,
               not_leaving_citytile=False,
               debug=False,
               max_future_turns=MAX_FUTURE_TURNS):
    self.game = game
    self.turn = game.turn
    self.game_map = game.game_map
    self.worker = worker
    self.not_leaving_citytile = not_leaving_citytile
    self.state_map = [[None
                       for h in range(self.game_map.height)]
                      for w in range(self.game_map.width)]
    self.debug = debug
    self.actions = []
    self.max_future_turns = max_future_turns

  @property
  def max_turns(self):
    return min(self.turn + self.max_future_turns, MAX_DAYS)

  @property
  def worker_upkeep(self):
    return get_unit_upkeep(self.worker)

  def wait_for_cooldown(self, state, extra_wait_days=0, debug=False):
    sim_turns = int(state.cooldown + extra_wait_days)
    if sim_turns < 1:
      # No need to wait.
      return state

    cargo = sim_on_cell(state.turn, state.cargo, self.worker.type, state.cell,
                        self.game, self.player, sim_turns)
    if cargo is None:
      # prt(f" - failed to wait_for_cooldown")
      return None

    next_state = SearchState(state.turn + sim_turns, state.cell, cargo,
                             state.cooldown + extra_wait_days - sim_turns)
    next_state.arrival_fuel = cargo_total_fuel(state.cargo)
    return next_state

  def move(self, state, next_cell):
    cooldown = get_unit_action_cooldown(self.worker)
    if is_night(state.turn + 1):
      cooldown *= 2
    is_citytile = cell_has_player_citytile(next_cell, self.game)
    if is_citytile:
      cooldown = 1  # 1 - 1 to get 0

    # Look ahead for cooldown to see whether worker will die on next move.
    # cargo = sim_on_cell(state.turn, state.cargo, self.worker.type,
    # next_cell, self.game, self.player, sim_turns=cooldown)
    # if cargo is None:
    # return None

    debug = False
    cargo = sim_on_cell(state.turn,
                        state.cargo,
                        self.worker.type,
                        next_cell,
                        self.game,
                        self.player,
                        sim_turns=1,
                        debug=debug)
    if cargo is None:
      return None

    next_state = SearchState(state.turn + 1, next_cell, cargo, cooldown - 1)
    next_state.arrival_fuel = cargo_total_fuel(state.cargo)
    return next_state

  @property
  def player(self):
    return self.game.players[self.worker.team]

  @property
  def opponent(self):
    return self.game.players[1 - self.worker.team]

  def compute(self):
    q = []

    def is_pushable_state(pos, arrival_turn):
      state = self.state_map[pos.x][pos.y]
      if state is None or arrival_turn < state.turn:
        return 1
      if arrival_turn == state.turn:
        return 0
      return -1

    def heap_push(prev_state, new_state):
      # Limit the search space to one circle.
      if new_state.turn > self.max_turns:
        return

      pos = new_state.pos
      state = self.state_map[pos.x][pos.y]
      if state is None or new_state < state:
        self.state_map[pos.x][pos.y] = new_state
        heapq.heappush(q, new_state)
        if prev_state:
          new_state.prev_pos = prev_state.pos
          new_state.prev_positions = [prev_state.pos]
        if state:
          state.deleted = True
        # if self.debug:
        # prt(f' push_head: prev{prev_state and prev_state.pos} to new{new_state.pos}, #prev={len(new_state.prev_positions)}')

      if state and state == new_state:
        state.prev_positions.append(prev_state.pos)

    def wait_then_move(cur_state, extra_wait_days=0, debug=False):
      # Waiting with cooldown.
      cur_state = self.wait_for_cooldown(cur_state,
                                         extra_wait_days=extra_wait_days,
                                         debug=debug)
      if cur_state is None:
        if debug:
          prt(f" - return from wait_for_cooldown")
        return

      assert cur_state.cooldown < 1
      for nb_cell in get_neighbour_positions(cur_state.pos,
                                             self.game_map,
                                             return_cell=True):
        pushable = is_pushable_state(nb_cell.pos, cur_state.turn + 1)
        if pushable < 0:
          continue

        # Can not go pass through enemy citytile.
        if cell_has_target_player_citytile(nb_cell, self.opponent):
          if debug:
            prt(f' skip move: {nb_cell.pos} has opponent citytile')
          continue

        # TODO: maybe not in cooldown?
        # Skip enemy unit in cooldown.
        if (nb_cell.unit and nb_cell.unit.team == self.opponent.team):
          # nb_cell.unit.team == self.game.opponent_id and not nb_cell.unit.can_act()):
          if debug:
            prt(f' skip move: {nb_cell.pos} has opponent unit')
          continue

        # if (nb_cell.unit and nb_cell.unit.team == self.player.team and not nb_cell.unit.can_act()):
        # continue

        next_state = self.move(cur_state, nb_cell)
        if not next_state:
          if debug:
            prt(f' skip move: cur_pos={cur_state.pos}, turn={cur_state.turn} to {nb_cell.pos}'
               )
          continue

        # Add extra waiting time to the moved state.
        next_state.extra_wait_days = extra_wait_days
        heap_push(cur_state, next_state)

    upkeep = get_unit_upkeep(self.worker)
    start_state = SearchState(self.turn, self.worker.cell, self.worker.cargo,
                              int(self.worker.cooldown))
    start_state.arrival_fuel = cargo_total_fuel(self.worker.cargo)

    heap_push(None, start_state)
    while q:
      cur_state = heapq.heappop(q)
      if cur_state.deleted:
        continue

      debug = False
      if (self.worker.id in DRAW_UNIT_LIST and
          cur_state.cell.pos in MAP_POS_LIST and
          self.not_leaving_citytile == False):
        debug = True
        # prt(f'target state matched: cur_state.pos={cur_state.pos}')

      # Do not leave citytile.
      is_player_citytile = cell_has_target_player_citytile(
          cur_state.cell, self.player)
      if (self.not_leaving_citytile and cur_state != start_state and
          is_player_citytile):
        if debug:
          prt(f' - continue')
        continue

      # if debug:
      # prt(f' wait_then_move (1): citytile={is_player_citytile}, night={is_night(cur_state.turn)}')
      wait_then_move(cur_state)

      # TODO: test wait in the days
      # Wait during the night.
      if is_player_citytile and is_night(cur_state.turn):
        cooldown_wait_days = int(cur_state.cooldown)
        days_till_next_day = CIRCLE_LENGH - (cur_state.turn % CIRCLE_LENGH)
        extra_wait_days = max(cooldown_wait_days,
                              days_till_next_day) - cooldown_wait_days
        if debug:
          prt(f' wait_then_move (2): cur_stat.turn={cur_state.turn},cd={cur_state.cooldown} extra_wait={extra_wait_days}, cooldown_wait_days={cooldown_wait_days}'
             )
        wait_then_move(cur_state, extra_wait_days, debug=debug)

    if self.debug:
      for y in range(self.game_map.height):
        for x in range(self.game_map.width):
          st = self.state_map[x][y]
          if st:
            a = annotate.text(x, y, f'{st.turn-self.turn}', fontsize=32)
            self.actions.append(a)

            a = annotate.text(st.pos.x,
                              st.pos.y,
                              f'{len(st.prev_positions)}',
                              fontsize=32)
            self.actions.extend([a])

            # prev = st.prev
            # if prev:
            # line = annotate.line(prev.pos.x, prev.pos.y,
            # st.pos.x, st.pos.y)
            # self.actions.append(line)
            for pos in st.prev_positions:
              if pos:
                line = annotate.line(pos.x, pos.y, st.pos.x, st.pos.y)
                self.actions.append(line)

  def get_dest_state(self, pos):
    return self.state_map[pos.x][pos.y]

  def query_dest_turns(self, pos):
    state = self.state_map[pos.x][pos.y]
    if state is None:
      return MAX_PATH_WEIGHT
    return state.turn - self.turn

  def get_next_step_path_points(self, target_pos, worker_pos):
    self.actions.clear()

    st = self.state_map[target_pos.x][target_pos.y]
    next_step_path_points = set()

    path_positions = {target_pos}
    q = deque([st])
    while q:
      st = q.popleft()
      if st is None:
        continue
      for prev_pos in st.prev_positions:
        prev = self.state_map[prev_pos.x][prev_pos.y]
        # Root
        if prev is None:
          continue

        # Collect points of next step from worker position.
        if prev.pos == worker_pos:
          if st.extra_wait_days == 0:
            next_step_path_points.add(st.pos)
          else:
            # Stay at current cell if the path require extra waiting time.
            next_step_path_points.add(prev.pos)

        if self.debug:
          line = annotate.line(prev.pos.x, prev.pos.y, st.pos.x, st.pos.y)
          self.actions.extend([line])

        # Added by other node.
        if prev.pos in path_positions:
          continue

        q.append(prev)
        path_positions.add(prev.pos)

    return next_step_path_points


class Clusetr:

  def __init__(self, cid, cells, game):
    self.cid = cid
    self.cells = cells
    self.game = game
    self.game_map = self.game.game_map
    # self._strategy = _strategy

  @property
  def any_cell(self):
    return self.cells[0]

  @property
  def size(self):
    return len(self.cells)

  @property
  @functools.lru_cache(maxsize=1, typed=False)
  def resource_type(self):
    return self.cells[0].resource.type

  @property
  @functools.lru_cache(maxsize=1, typed=False)
  def total_fuel(self):
    return sum(resource_fuel(c.resource) for c in self.cells)

  @property
  @functools.lru_cache(maxsize=1, typed=False)
  def boundary_positions(self):
    """boundary positions:
    1) not a resoruce cell
    2) resource cell, but not the same type as cluster type."""
    positions = set()
    for cluster_cell in self.cells:
      for nb_cell in get_neighbour_positions(cluster_cell.pos,
                                             self.game_map,
                                             return_cell=True):
        if (not nb_cell.has_resource() or
            nb_cell.resource.type != self.resource_type):
          positions.add(nb_cell.pos)
    return positions

  @property
  @functools.lru_cache(maxsize=1, typed=False)
  def resource_positions(self):
    return {c.pos for c in self.cells}

  def is_arrived(self, pos):
    return pos in self.boundary_positions or pos in self.resource_positions

  @functools.lru_cache(maxsize=2, typed=False)
  def get_open_boundary_positions(self, can_build=False):
    """open positions:
    1) not a citytile AND not opponent unit
    2) (can build) not resource tile."""
    open_positions = set()
    for pos in self.boundary_positions:
      boundary_cell = self.game_map.get_cell_by_pos(pos)
      if (boundary_cell.citytile is None and
          (boundary_cell.unit is None or
           boundary_cell.unit.team != self.game.opponent.team)):
        if can_build and boundary_cell.resource != None:
          continue
        open_positions.add(boundary_cell.pos)
    return open_positions

  def worker_cluster_turns_info(self, worker, debug=False):
    """Returns the number of turns to arrive at the cluster and the arrival
    position."""
    # If woker already on boundary or cluster resource cells.
    if (worker.pos in self.boundary_positions or
        worker.pos in self.resource_positions):
      return 0, [worker.pos]

    best_positions = []
    best_arrival_turns = MAX_PATH_WEIGHT

    # TODO: use boundary + resource_positions
    open_positions = (self.get_open_boundary_positions() |
                      self.resource_positions)
    for pos in open_positions:
      quick_path, arrival_turns = _strategy.select_quicker_path(worker, pos)

      if arrival_turns < best_arrival_turns:
        best_positions = [pos]
        best_arrival_turns = arrival_turns
      elif arrival_turns == best_arrival_turns:
        best_positions.append(pos)
    return best_arrival_turns, best_positions

  @property
  @functools.lru_cache(maxsize=1, typed=False)
  def player_citytile_count(self):
    cnt = 0
    for pos in self.boundary_positions:
      cell = self.game_map.get_cell_by_pos(pos)
      if cell_has_player_citytile(cell, self.game):
        cnt += 1
    return cnt


class ClusterInfo:

  def __init__(self, game):
    self.game = game
    self.game_map = game.game_map
    self.position_to_cid = np.ones(
        (self.game_map.width, self.game_map.height), dtype=np.int64) * -1
    self.max_cluster_fuel = 0
    self.max_cluster_id = 0
    self.clusters = {}

  def set_cid(self, pos, cid):
    self.position_to_cid[pos.x][pos.y] = cid

  def get_cid(self, pos):
    return self.position_to_cid[pos.x][pos.y]

  def c(self, cid):
    return self.clusters[cid]

  def get_clusters(self, cluster_ids):
    for cid in cluster_ids:
      yield self.c(cid)

  def get_cluster(self, cid):
    return self.clusters[cid]

  @functools.lru_cache(maxsize=512)
  def get_neighbour_cells_cluster_ids(self, near_resource_tile):
    cluster_ids = set()
    for nb_cell in get_neighbour_positions(near_resource_tile.pos,
                                           self.game_map,
                                           return_cell=True):
      newpos = nb_cell.pos
      cid = self.get_cid(newpos)
      if cid >= 0:
        cluster_ids.add(cid)
    return cluster_ids

  def count_min_boundary_near_resource_tiles(self, near_resource_tile):
    n_open, n_boundary = 999, 999
    cluster_ids = self.get_neighbour_cells_cluster_ids(near_resource_tile)
    for cid in cluster_ids:
      cluster = self.clusters[cid]
      cluster_type = cluster.resource_type
      if cluster_type == Constants.RESOURCE_TYPES.WOOD:
        continue

      boundary_positions = cluster.boundary_positions
      open_positions = cluster.get_open_boundary_positions()

      n_open = min(n_open, len(open_positions))
      n_boundary = min(n_boundary, len(boundary_positions))
    return n_boundary, n_open

  def cell_next_to_target_cluster(self, worker, near_resource_tile):
    if worker.target_cluster_id < 0:
      return False
    cluster_ids = self.get_neighbour_cells_cluster_ids(near_resource_tile)
    return worker.target_cluster_id in cluster_ids

  def cluster(self):

    def search_cluster(start_cell):
      q = deque([start_cell.pos])
      self.set_cid(start_cell.pos, max_cluster_id)

      while q:
        cur = q.popleft()
        cur_cell = self.game_map.get_cell_by_pos(cur)
        for nb_cell in get_neighbour_positions(cur,
                                               self.game_map,
                                               return_cell=True):
          newpos = nb_cell.pos

          if not nb_cell.has_resource():
            continue

          # Split different resource into different cluster
          if cur_cell.resource.type != nb_cell.resource.type:
            continue

          cid = self.get_cid(newpos)
          if cid >= 0:
            continue

          self.set_cid(newpos, max_cluster_id)
          q.append(nb_cell.pos)

    max_cluster_id = 0
    for x in range(self.game_map.width):
      for y in range(self.game_map.height):
        pos = Position(x, y)
        if self.get_cid(pos) >= 0:
          continue

        cell = self.game_map.get_cell_by_pos(pos)
        if not cell.has_resource():
          continue

        search_cluster(cell)
        max_cluster_id += 1
    self.max_cluster_id = max_cluster_id

    # Construct cluster objects.
    for cid in range(self.max_cluster_id):
      x_pos, y_pos = np.where(self.position_to_cid == cid)
      cluster_cells = [
          self.game_map.get_cell_by_pos(Position(x, y))
          for x, y in zip(x_pos, y_pos)
      ]
      cluster = Clusetr(cid, cluster_cells, self.game)
      self.clusters[cid] = cluster

      self.max_cluster_fuel = max(self.max_cluster_fuel, cluster.total_fuel)
    prt(f't={self.game.turn}, total_cluster={max_cluster_id}', file=sys.stderr)

  def query_cluster_fuel_factor(self, pos):
    cid = self.position_to_cid[pos.x][pos.y]
    if cid < 0:
      return 0
    return self.get_cluster_fuel_factor(cid)

  def get_cluster_fuel_factor(self, cid):
    if cid < 0:
      return 0
    c = self.clusters[cid]
    return (c.total_fuel / self.max_cluster_fuel)

  def get_cluster_type(self, cid):
    return self.c(cid).resource_type

  @functools.lru_cache(maxsize=1024)
  def cell_has_player_citytile_on_target_cluster(self, worker,
                                                 near_resource_tile):
    if worker.target_cluster_id < 0:
      return False

    for nb_cell in get_neighbour_positions(near_resource_tile.pos,
                                           self.game_map,
                                           return_cell=True):
      cid = self.get_cid(nb_cell.pos)
      if cid != -1 and cid != worker.target_cluster_id:
        n_citytile = self.c(cid).player_citytile_count
        if n_citytile > 0:
          return True
    return False

  @functools.lru_cache(maxsize=1023, typed=False)
  def get_min_cluster_arrival_turns_for_opponent_unit(self, cid, unit):
    min_pos = None
    min_dist = MAX_PATH_WEIGHT

    cluster = self.c(cid)
    for cell in cluster.cells:
      cluster_pos = cell.pos

      shortest_path, _ = _strategy.quickest_path_pairs[unit.id]
      dist = shortest_path.shortest_dist(cluster_pos)
      if dist < min_dist:
        min_dist = dist
        min_pos = cluster_pos

    if min_dist == MAX_PATH_WEIGHT:
      return MAX_PATH_WEIGHT, None
    min_turns = unit_arrival_turns(self.game.turn, unit, min_dist)
    return min_turns, min_pos

  # TODO: is threat_dist=2 a good choice?
  @functools.lru_cache(maxsize=512)
  def get_opponent_unit_nearest_cluster_ids(self,
                                            unit,
                                            threat_turns=4,
                                            debug=False):
    """Returns the nearest cluster that
    1) opponent can collect fuel (researched)
    2) with no enemy city.
    """
    min_turns = MAX_PATH_WEIGHT
    cluster_ids = set()
    threat_cluster_ids = set()
    if self.game.is_night:
      threat_turns *= 2
    for cid in range(self.max_cluster_id):
      turns, cluster_pos = self.get_min_cluster_arrival_turns_for_opponent_unit(
          cid, unit)
      if cluster_pos is None:
        continue

      cluster_cell = self.game_map.get_cell_by_pos(cluster_pos)
      if not is_resource_researched(
          cluster_cell.resource, self.game.opponent, move_days=turns):
        continue

      if threat_turns and turns <= threat_turns:
        threat_cluster_ids.add(cid)

      # if debug:
      # prt(f" *** >> cid={cid}, min_turns={turns}, cluster_pos={cluster_pos}")

      if turns < min_turns:
        cluster_ids.clear()
        cluster_ids.add(cid)
        min_turns = min(min_turns, turns)
      elif turns == min_turns:
        cluster_ids.add(cid)

    # if debug:
    # prt(f" > oppo_unit={unit.id}, near_cluster_ids={cluster_ids}, threat_cluster_ids={threat_cluster_ids}")
    # for cid in cluster_ids | threat_cluster_ids:
    # prt(f"  cid={cid}, cell={self.c(cid).any_cell.pos}")
    return cluster_ids | threat_cluster_ids

  @functools.lru_cache(maxsize=1023, typed=False)
  def get_min_turns_to_cluster_near_resource_cell_for_opponent_unit(
      self, cid, unit):
    min_dist = MAX_PATH_WEIGHT
    cluster = self.c(cid)
    open_positions = cluster.get_open_boundary_positions()
    for pos in open_positions:
      shortest_path, _ = _strategy.quickest_path_pairs[unit.id]
      dist = shortest_path.shortest_dist(pos)
      if dist < min_dist:
        min_dist = dist

    if min_dist == MAX_PATH_WEIGHT:
      return MAX_PATH_WEIGHT
    min_turns = unit_arrival_turns(self.game.turn, unit, min_dist)
    return min_turns


class Strategy:

  def __init__(self):
    self.actions = []
    self.game = LuxGame()
    self.accepted_transfer_offers = {}
    self.fuel_city_by_transfer_positions = {}
    self.is_wood_city_tile = set()
    self.blacklist_city_building_positions = set()
    self.non_wood_resource_locations = set()
    self.idle_woker_ids = set()

  @property
  def circle_turn(self):
    return self.game.circle_turn

  @property
  def day_factor(self):
    return 1 - self.game.days_this_round / DAY_LENGTH

  @property
  def game_map(self):
    return self.game.game_map

  @property
  def player(self):
    return self.game.player

  def player_available_workers(self):
    workers = [
        unit for unit in self.game.player.units
        if (unit.is_worker() and not unit.has_planned_action)
    ]
    return workers

  def update(self, observation, configuration):
    self.game.update(observation, configuration)

    # Clear up actions for current step.
    self.actions = []

  def add_unit_action(self, unit, action):
    assert unit.has_planned_action == False

    unit.has_planned_action = True
    self.actions.append(action)

  @functools.lru_cache(maxsize=4096, typed=False)
  def select_quicker_path(self, worker, target_pos):
    path, path_wo_cc = self.quickest_path_pairs[worker.id]
    dest_turns = path.query_dest_turns(target_pos)

    quicker_path = path
    quicker_dest_turns = dest_turns

    dest_turns_wo_cc = path_wo_cc.query_dest_turns(target_pos)
    if dest_turns_wo_cc < quicker_dest_turns:
      quicker_path = path_wo_cc
      quicker_dest_turns = dest_turns_wo_cc
    return quicker_path, quicker_dest_turns

  def update_game_map_info(self):
    # TODO: ref citytile to city
    self.has_collectable_resource_on_map = False
    for y in range(self.game.map_height):
      for x in range(self.game.map_width):
        cell = self.game_map.get_cell(x, y)
        cell.unit = None
        cell.has_buildable_neighbour = False
        cell.units = []

        cell.is_coal_target = False
        cell.is_uranium_target = False
        if cell.has_resource():
          cell.is_coal_target = is_resource_coal(cell.resource)
          cell.is_uranium_target = is_resource_uranium(cell.resource)

        cell.is_near_resource = self.is_near_resource_cell(cell)
        cell.n_citytile_neighbour = self.count_citytile_neighbours(cell)

        # This is a flag for calling for idle worker to collect fuel for city.
        cell.is_first_citytile = False

        if self.game.turn == 0:
          if cell.is_coal_target or cell.is_uranium_target:
            self.non_wood_resource_locations.add(cell.pos)

        if (cell.has_resource() and
            is_resource_researched(cell.resource, self.player)):
          self.has_collectable_resource_on_map = True

        # if cell.pos in MAP_POS_LIST:
        # prt(f"cell.pos={cell.pos}, has_res={cell.has_resource()} res={cell.resource}, coal={cell.is_coal_target}, urnaium={cell.is_uranium_target}")

    # if self.game.turn == 0:
    # prt(f'number of non_wood_resource_locations: {len(self.non_wood_resource_locations)}, {self.non_wood_resource_locations}')
    # pass

    for unit in self.game.player.units:
      cell = self.game_map.get_cell_by_pos(unit.pos)
      cell.unit = unit
      cell.units.append(unit)

    for unit in self.game.opponent.units:
      cell = self.game_map.get_cell_by_pos(unit.pos)
      cell.unit = unit
      cell.units.append(unit)

  # @timeit
  def update_unit_info(self):
    self.quickest_path_pairs = {}
    n_unit = len(self.game.player.units)
    for i, unit in enumerate(self.game.player.units):
      unit.cell = self.game_map.get_cell_by_pos(unit.pos)
      unit.has_planned_action = False
      unit.target_pos = unit.pos
      unit.target = self.game_map.get_cell_by_pos(unit.pos)
      unit.target_score = 0

      unit.transfer_build_locations = set()
      unit.target_cluster_id = -1
      unit.target_cluster = None
      unit.is_cluster_owner = False
      unit.is_transfer_worker = False
      unit.target_city_id = None

      # Flag the unit if after the worker has not task after initial planning.
      unit.is_idle_worker = False

      unit.unit_night_count = cargo_night_endurance(unit.cargo,
                                                    get_unit_upkeep(unit))

      left_turns_this_round = get_left_turns_this_round(self.game.turn)

      # TODO: it should not be used?
      # surviving_turns is only based on current cargo, map info is not included.
      surviving_turns = unit_surviving_turns(self.game.turn, unit)
      unit.is_cargo_not_enough_for_nights = surviving_turns < left_turns_this_round

      unit.is_carrying_coal_or_uranium = (unit.cargo.coal > 0 or
                                          unit.cargo.uranium > 0)

      debug = False
      debug = (unit.id in DRAW_UNIT_LIST and DRAW_QUICK_PATH_VALUE)

      future_turns = CIRCLE_LENGH
      if unit.id in self.idle_woker_ids:
        future_turns = CIRCLE_LENGH + 20
      quickest_path_wo_citytile = QuickestPath(self.game,
                                               unit,
                                               not_leaving_citytile=True,
                                               debug=debug,
                                               max_future_turns=future_turns)
      quickest_path_wo_citytile.compute()

      quickest_path = quickest_path_wo_citytile

      # TODO: could use a larger threshold
      if unit.is_carrying_coal_or_uranium or unit.get_cargo_space_left() == 0:
        quickest_path = quickest_path_wo_citytile
      else:
        if ((self.game_map.width < 32 or n_unit < 50 or
             i < MAX_UNIT_NUM - n_unit)):
          quickest_path = QuickestPath(self.game,
                                       unit,
                                       max_future_turns=future_turns,
                                       debug=debug)
          quickest_path.compute()
          self.actions.extend(quickest_path.actions)

      self.quickest_path_pairs[unit.id] = (quickest_path,
                                           quickest_path_wo_citytile)

      unit.cid_to_tile_pos = {}
      unit.cid_to_cluster_turns = {}

    for unit in self.game.opponent.units:
      shortest_path = ShortestPath(self.game, unit, ignore_unit=True)
      shortest_path.compute()
      self.quickest_path_pairs[unit.id] = (shortest_path, None)

    # Add first city as wood city
    if self.game.turn == 0:
      assert len(self.game.player.units) == 1
      unit = self.game.player.units[0]
      assert cell_has_player_citytile(unit.cell, self.game)
      self.is_wood_city_tile.add(unit.pos)

  @functools.lru_cache(maxsize=1024, typed=False)
  def get_nearest_opponent_unit_to_cell(self, cell, debug=False):
    """The shortest path dist is capped at a max dist of 6."""
    min_dist = MAX_PATH_WEIGHT
    min_unit = None
    for unit in self.game.opponent.units:
      shortest_path, _ = self.quickest_path_pairs[unit.id]
      dist = shortest_path.shortest_dist(cell.pos)

      # Use `distance_to` as a proxy
      if dist >= MAX_PATH_WEIGHT:
        dist2 = unit.pos.distance_to(cell.pos)
        if dist2 >= ShortestPath.MAX_SEARCH_DIST:
          dist = dist2

      # if debug and min_unit:
      # prt(f' > c={cell.pos}, oppo={unit.id}@{unit.pos} dist={dist}')
      if dist < min_dist:
        min_dist = dist
        min_unit = unit

    # if debug and min_unit:
    # prt(f' c={cell.pos}, nearest oppo {min_unit.pos} dist={min_dist}')

    # opponent units could be empty
    min_turns = MAX_PATH_WEIGHT
    if min_unit:
      min_turns = unit_arrival_turns(self.game.turn, min_unit, min_dist)
    return min_turns, min_unit

  def count_citytile_neighbours(self, cell, min_citytile=1):
    cnt = 0
    for nb_cell in get_neighbour_positions(cell.pos,
                                           self.game_map,
                                           return_cell=True):
      if cell_has_player_citytile(nb_cell, self.game):
        cnt += 1
    return cnt

  def is_near_resource_cell(self, cell):

    def has_resource_tile_neighbour(cell):
      for nb_cell in get_neighbour_positions(cell.pos,
                                             self.game_map,
                                             return_cell=True):
        if nb_cell.has_resource():
          return True
      return False

    return (not cell.has_resource() and cell.citytile is None and
            has_resource_tile_neighbour(cell))

  @timeit
  def assign_worker_target(self, workers, plan_idx=0):
    g = self.game
    player = g.player

    def is_deficient_resource_tile(resource_tile):
      if not is_night(self.game.turn):
        return False

      if not resource_tile.has_resource():
        return True

      if (is_resource_wood(resource_tile.resource) and
          resource_tile.resource.amount < 60):
        return True
      return False

    def is_resource_tile_can_save_dying_worker(resource_tile,
                                               worker,
                                               debug=False):
      # TODO: remove is_worker_on_last_city_tiles? so worker can go out
      if (is_worker_on_last_city_tiles(worker) or
          is_deficient_resource_tile(resource_tile)):
        return False

      # Can't reach this resource/near tile.
      quick_path, arrival_turns = self.select_quicker_path(
          worker, resource_tile.pos)
      if arrival_turns >= MAX_PATH_WEIGHT:
        return False

      if worker.is_cargo_not_enough_for_nights:
        surviving_turns = get_surviving_turns_at_cell(worker,
                                                      quick_path,
                                                      resource_tile,
                                                      debug=debug)
        cell_nights = estimate_cell_night_count(worker,
                                                resource_tile,
                                                g.game_map,
                                                arrival_turns,
                                                surviving_turns,
                                                debug=debug)
        round_nights = get_night_count_this_round(self.game.turn)
        if debug:
          print(
              f' > unit_night_count={worker.unit_night_count}, arrival_turns={arrival_turns}, '
              f'cell_nights={cell_nights}, round_nights={round_nights}')
        if worker.unit_night_count + cell_nights >= round_nights:
          return True
      return False

    def is_worker_on_last_city_tiles(worker):
      citytile = worker.cell.citytile
      if citytile is None:
        return False
      city = self.player.cities[citytile.cityid]
      city_last = not city_wont_last_at_nights(self.game.turn, city)
      return True

    MAX_WEIGHT_VALUE = 10000
    CLUSTER_BOOST_WEIGHT = 200 * 40
    UNIT_SAVED_BY_RES_WEIGHT = 10

    DEFAULT_RESOURCE_WT = 1.2

    def get_resource_weight(worker, resource_tile, arrival_turns, quick_path):
      # Give a small weight for any resource 0.1 TODO: any other option?
      wt = 0

      fuel_wt_type = 0
      if is_deficient_resource_tile(resource_tile):
        # do not goto resource tile at night if there is not much.
        fuel_wt = 0
        fuel_wt_type = 'deficient'
      elif (is_resource_wood(resource_tile.resource) and
            (not resource_tile.has_buildable_neighbour
             or worker.cargo.wood >= 20)):
        # 1) For wood cell with no buildable neightbor, demote its weight
        # 2) For worker with wood >= 40, move build need 5 turns, while wait need only 4.

        # Use surviving_turns at the arrival state.
        surviving_turns = get_surviving_turns_at_cell(worker, quick_path,
                                                      resource_tile)

        # Use cell value as a way of demote weighting.
        _, fuel_wt = get_cell_resource_values(
            resource_tile,
            player,
            # unit=worker,
            move_days=arrival_turns,
            surviving_turns=surviving_turns)
        fuel_wt /= 2
        fuel_wt_type = ('wood_not_buildable' if (not resource_tile.has_buildable_neighbour)
                        else 'worker_wood>=20')
      else:
        debug = False
        if worker.id in DRAW_UNIT_LIST and resource_tile.pos in MAP_POS_LIST:
          debug = True

        # Use surviving_turns at the arrival state.
        surviving_turns = get_surviving_turns_at_cell(worker, quick_path,
                                                      resource_tile)
        _, fuel_wt = get_one_step_collection_values(
            resource_tile,
            player,
            self.game,
            # unit=worker,
            move_days=arrival_turns,
            surviving_turns=surviving_turns,
            debug=debug)
        fuel_wt_type = 'normal'

      default_res_wt = 0
      if fuel_wt:
        default_res_wt = DEFAULT_RESOURCE_WT / 2

      #TODO: encourage worker into near/resource tile, not necessary dying
      # Try to hide next to resource grid in the night.
      if is_resource_tile_can_save_dying_worker(resource_tile, worker):
        wt += UNIT_SAVED_BY_RES_WEIGHT

      # TODO: Consider drop cluster boosting when dist <= 1
      cid = self.cluster_info.get_cid(resource_tile.pos)
      boost_cluster = 0
      if (worker.is_cluster_owner and worker.target_cluster_id == cid):
        if not self.ci.c(cid).is_arrived(worker.pos):
          cluster_fuel_factor = self.cluster_info.get_cluster_fuel_factor(cid)
          boost_cluster += CLUSTER_BOOST_WEIGHT * cluster_fuel_factor

      debug = False
      if worker.id in DRAW_UNIT_LIST and resource_tile.pos in MAP_POS_LIST:
        debug = True

      opponent_weight = 0
      # Test can collect weight first (ignore can not mine cell)
      oppo_arrival_turns = MAX_PATH_WEIGHT
      if fuel_wt > 0:
        oppo_arrival_turns, nearest_oppo_unit = self.get_nearest_opponent_unit_to_cell(
            resource_tile)

        # Use a small weight to bias the position towards opponent's positions.
        if nearest_oppo_unit:
          if oppo_arrival_turns < MAX_PATH_WEIGHT:
            opponent_weight += 1e-3 / dd(oppo_arrival_turns)

        # Use a large weight to defend my resource
        # 0) opponent unit is near this tile
        # 1) worker can arrive at this cell quicker than opponent
        # 2) cluster id of tile is opponent unit 's nearest cluster
        # 3) this cell is the nearest one in cluster to the opponent unit.
        if (nearest_oppo_unit and
            oppo_arrival_turns <= MIN_DEFEND_ENEMY_ARRIVAL_TRUNS and
            arrival_turns <= oppo_arrival_turns and
            (cid in self.ci.get_opponent_unit_nearest_cluster_ids(
                nearest_oppo_unit)) and
            (self.ci.get_min_cluster_arrival_turns_for_opponent_unit(
                cid, nearest_oppo_unit)[0] == oppo_arrival_turns)):
          boost = (101 if arrival_turns < oppo_arrival_turns else 11)
          opponent_weight += boost

      demote_opponent_unit = 0
      # if self.has_can_act_opponent_unit_as_neighbour(resource_tile):
      # demote_opponent_unit = -0.1

      v = ((wt) / dd(arrival_turns) + boost_cluster + fuel_wt +
           opponent_weight + demote_opponent_unit + default_res_wt)
      if worker.id in DRAW_UNIT_LIST and resource_tile.pos in MAP_POS_LIST:
        prt(f"w[{worker.id}] v={v}, res={resource_tile.pos} 4. wt={wt}, boost_cluster={boost_cluster}, fuel_wt={fuel_wt}, opponent_weight={opponent_weight}, min_oppo_arrival_turns={oppo_arrival_turns}"
            f" not_leave_city={quick_path.not_leaving_citytile}, demote_oppo_unit={demote_opponent_unit}, default_res_wt={default_res_wt}"
           )
      return v

    @functools.lru_cache(maxsize=1024)
    def worker_city_min_arrival_turns(worker, city):
      _, quick_path = self.quickest_path_pairs[worker.id]
      min_arrival_turns = MAX_PATH_WEIGHT
      for citytile in city.citytiles:
        turns = quick_path.query_dest_turns(citytile.pos)
        min_arrival_turns = min(min_arrival_turns, turns)
      return min_arrival_turns

    CITYTILE_LOST_WEIGHT = 1000

    def get_city_tile_weight(worker, city_cell, arrival_turns):
      """
      1. collect fuel (to keep powering citytile)
      2. protect dying worker [at night]: to dying woker
      3. receive fuel from a fuel full worker on cell: get help from worker.
      """
      citytile = city_cell.citytile
      city = g.player.cities[citytile.cityid]

      wt = 0

      # [1] Stay at city will gain this amout of fuel
      # amount, fuel = get_one_step_collection_values(citytile.cell, player,
      # self.game)
      # if self.game.is_night:
      # wt += max(fuel - LIGHT_UPKEEP['CITY'], 0)
      # wt += city_last_nights(city)

      #TODO: add property to city
      # [2] Try to hide in the city if worker will run out of fuel at night
      city_wont_last = city_wont_last_at_nights(g.turn, city)
      city_last = not city_wont_last
      if worker.is_cargo_not_enough_for_nights:
        # Hide in the city
        if city_last:
          wt += max(min(city_last_nights(city) / 10, 1),
                    0) * 0.5 * self.day_factor
        # elif cargo_total_amount(worker.cargo) == 0:
        # Escape from dying city.
        # wt = -99999
        # return wt

      # TODO(wangfei): estimate worker will full
      n_citytile = len(city.citytiles)

      _, quick_path = self.quickest_path_pairs[worker.id]
      arrival_turns_wo_city = quick_path.query_dest_turns(city_cell.pos)

      city_last_turns = city.last_turns
      min_city_arrival_turns = worker_city_min_arrival_turns(worker, city)

      city_crash_boost = 0
      city_survive_boost = 0

      # TODO: tmp disalbe it.
      # If the worker is a wood full worker, goto nearest city tiles when possible.
      # if (city_wont_last and arrival_turns <= city_last_turns and
          # worker.cargo.wood == WORKER_RESOURCE_CAPACITY and
          # not worker.is_cluster_owner):
        # wt += 1

      # If a worker can arrive at this city with some min fuel (or full cargo)
      unit_fuel = cargo_total_fuel(worker.cargo)
      city_crash_boost_loc = ''
      if city_wont_last:
        _, quick_path = self.quickest_path_pairs[worker.id]
        arrival_turns_wo_city = quick_path.query_dest_turns(city_cell.pos)
        dest_state = quick_path.state_map[city_cell.pos.x][city_cell.pos.y]
        if (arrival_turns_wo_city ==
            min_city_arrival_turns  # only goto nearest city tiles.
            and arrival_turns_wo_city <=
            city_last_turns):  # city should last when arrived.
          not_full_woker_goto_city = (
              city_last_turns - arrival_turns_wo_city <=
              6  # do not goto city too earlier.
              and dest_state.arrival_fuel >= 60 and unit_fuel >= 60)
          full_worker_goto_city = (worker.get_cargo_space_left() == 0 and
                                   worker.is_carrying_coal_or_uranium)
          if (not_full_woker_goto_city or full_worker_goto_city):
            is_wood_city = city_cell.pos in self.is_wood_city_tile
            if n_citytile <= 1 or is_wood_city:
              city_crash_boost += 1
              # city_crash_boost = 1 - int(is_wood_city)
              city_crash_boost_loc = 'wood_city_crash'
            else:
              city_crash_boost += worker_total_fuel(worker) * n_citytile
              # city_crash_boost += n_citytile * max(CITYTILE_LOST_WEIGHT,
              # worker_total_fuel(worker))
              city_crash_boost_loc = 'other_city_crash'

      # Also limit resource to next day.
      if (plan_idx > 0 and
          (worker.id, city.id) not in self.keep_city_alive_till_next_day and
          city.city_next_day_fuel_req < 0):
        city_crash_boost = 0

      def is_worker_ready_for_delivery(worker, city_cell):
        """
        > coal: 300 = 10 (fuel_rate) * 5 (collect_rate) * 6 days
        > uranium: 480 = 40 (fuel_rate) * 2 (collect_rate) * 6 days
        """
        _, quick_path = self.quickest_path_pairs[worker.id]
        dest_state = quick_path.state_map[city_cell.pos.x][city_cell.pos.y]
        if dest_state is None:
          return False

        c = worker.cargo
        if cargo_total_amount(c) > 85 and dest_state.arrival_fuel > 85:
          return True

        fuel = cargo_total_fuel(c)
        if c.uranium > 0 and fuel >= 480 and dest_state.arrival_fuel >= 480:
          return True
        return fuel >= 300 and dest_state.arrival_fuel >= 300

      def no_resource_to_explore(min_wait_turns=10):
        if self.has_collectable_resource_on_map:
          return False

        player = self.game.player
        if not player.researched_coal():
          coal = Resource(Constants.RESOURCE_TYPES.COAL, 1)
          return not is_resource_researched(
              coal, player, move_days=min_wait_turns)
        if not player.researched_uranium():
          uranium = Resource(Constants.RESOURCE_TYPES.URANIUM, 1)
          return not is_resource_researched(
              uranium, player, move_days=min_wait_turns)
        return True

      city_survive_boost_loc = ''

      is_worker_with_enough_resource = is_worker_ready_for_delivery(
          worker, city_cell)
      is_idle_worker = (plan_idx > 1 and worker.is_idle_worker)
      is_nearest_city_tile = arrival_turns_wo_city == min_city_arrival_turns
      not_city_survive_till_end = (self.game.turn + city_last_turns) < MAX_DAYS

      is_wood_city = city_cell.pos in self.is_wood_city_tile
      if (unit_fuel > 0 and is_nearest_city_tile and
          (is_worker_with_enough_resource or is_idle_worker or
           no_resource_to_explore()) and not_city_survive_till_end):

        surviving_rate = city_last_turns / (MAX_DAYS - self.game.turn + 1)
        if city_wont_last:
          surviving_rate = 0

        decay = 1 if city_wont_last else 2.5
        fuel = worker_total_fuel(worker)
        if n_citytile <= 1 or is_wood_city:
          fuel = 1
          # fuel = (1 - int(is_wood_city))

        city_survive_boost += (1 - surviving_rate) * n_citytile * fuel / decay

      # Ignore |city_survive_boost| after plan=0 if city already receive enough fuel
      # from the plan=0.
      if (plan_idx > 0 and
          (worker.id, city.id) not in self.keep_city_alive_till_game_end and
          city.city_game_end_fuel_req < 0):
        city_survive_boost = 0

      if city_crash_boost > 0 or city_survive_boost > 0:
        self.worker_fuel_city_tasks.add((worker.id, city_cell.pos))

      # Boost when worker has resource and city tile won't last.
      days_left = DAY_LENGTH - self.circle_turn
      round_nights = get_night_count_this_round(g.turn)

      # [not used] Boost based on woker, city assignment.
      if (worker.target_city_id == citytile.cityid and
          worker.pos.distance_to(city_cell.pos) == 1):
        wt += 1000 * n_citytile

      # When there is no fuel on the map, go back to city
      # if self.cluster_info.max_cluster_fuel == 0:
      # wt += 100

      # Overwrite arrival_turns
      if self.is_worker_fuel_city_task(worker, city_cell.pos):
        _, quick_path = self.quickest_path_pairs[worker.id]
        arrival_turns = quick_path.query_dest_turns(city_cell.pos)
        if arrival_turns >= MAX_PATH_WEIGHT:
          return -9999

      # Assume transfer happens late in the game, so all worker can help?
      # * Only boost for one unit.
      # * Prioritize unit in the same city.
      receive_transfer_wt = 0
      if (city_cell.is_first_citytile and worker_total_fuel(worker) <= 20 and
          citytile.pos in self.fuel_city_by_transfer_positions):
        receive_transfer_wt = 1000 + self.fuel_city_by_transfer_positions[
            citytile.pos]

      v = (wt / dd(arrival_turns) +
           (city_crash_boost / dd(arrival_turns, r=1.10)) +
           (city_survive_boost / dd(arrival_turns, r=1.10)) +
           (receive_transfer_wt / dd(arrival_turns)))
      if (city_cell.is_first_citytile and worker.id in DRAW_UNIT_LIST and
          city_cell.pos in MAP_POS_LIST):
        prt(f"ccw-4: plan[{plan_idx}] {worker.id} tar={worker.target.pos} => city={city_cell.pos}  v={v}, wt={wt}, city_crash={city_crash_boost}@[{city_crash_boost_loc}],"
            f"city_survive={city_survive_boost}@[{city_survive_boost_loc}]"
            f"recv={receive_transfer_wt}, city_last_turns={city.last_turns}, arrival_turns={arrival_turns}"
            f" no_resource_to_explore={no_resource_to_explore()}, wood_city={is_wood_city}"
           )
        # prt city info
        # prt(f"city={city.id}, f={city.fuel}, keep={city.light_upkeep}, last_turns={city.last_turns}, nights={city_last_nights(city)}")
      return v

    MIN_DEFEND_ENEMY_ARRIVAL_TRUNS = 8

    # TODO(wangfei): merge near resource tile and resource tile weight functon
    def get_near_resource_tile_weight(worker, near_resource_tile, arrival_turns,
                                      quick_path):
      debug = False
      if worker.id in DRAW_UNIT_LIST and near_resource_tile.pos in MAP_POS_LIST:
        debug = True
      wt = 0
      # Not using unit so near resource tile will have more weight over resource tile.
      surviving_turns = get_surviving_turns_at_cell(worker, quick_path,
                                                    near_resource_tile)
      amount, fuel_wt = get_one_step_collection_values(
          near_resource_tile,
          player,
          self.game,
          # unit=worker,
          move_days=arrival_turns,
          surviving_turns=surviving_turns,
          debug=debug)
      # TODO: use amount > 0
      is_fuelable_near_resource_tile = bool(fuel_wt > 0)

      # fuel_wt /= 3  # Use smaller weight for near resource tile

      # TODO: should consider wait time
      default_res_wt = 0
      if is_fuelable_near_resource_tile > 0:
        default_res_wt = DEFAULT_RESOURCE_WT

      # Boost the target collect amount by 2 (for cooldown) to test for citytile building.
      # it's 2, because one step move and one step for cooldown
      # Only build at fuelabe near resource tile.
      build_city_bonus = False
      build_city_bonus_off_reason = '-'
      days_left = BUILD_CITYTILE_ROUND - self.circle_turn
      if (is_fuelable_near_resource_tile and
          worker_enough_cargo_to_build(worker, amount * 2) and
          days_left >= arrival_turns):
        build_city_bonus = f'build_near_resource_tile'

      # To build on transfer build location.
      # prt(f'w[{worker.id}], transfer_build_locations={worker.transfer_build_locations}')
      transfer_build_wt = get_boost_transfer_build_weight(
          worker, near_resource_tile)
      is_transfer_build_position = (transfer_build_wt > 0)
      if is_transfer_build_position:
        build_city_bonus = 'transfer_build'

      boost_cluster = 0
      is_next_to_target_cluster = self.ci.cell_next_to_target_cluster(
          worker, near_resource_tile)
      if is_next_to_target_cluster:
        cid = worker.target_cluster_id
        if not self.ci.c(cid).is_arrived(worker.pos):
          cluster_fuel_factor = self.ci.get_cluster_fuel_factor(cid)
          boost_cluster += CLUSTER_BOOST_WEIGHT * cluster_fuel_factor

      # TODO: decide upon front, which path to use: cargo > 0?
      # Why it is needed? because unit need to be there with full cargo.
      # Overwrite arrival_turns or not boost city building.
      if (self.is_worker_building_citytile(worker, near_resource_tile.pos) and
          not is_transfer_build_position):
        _, quick_path = self.quickest_path_pairs[worker.id]
        tmp_arrival_turns = quick_path.query_dest_turns(near_resource_tile.pos)
        if tmp_arrival_turns >= MAX_PATH_WEIGHT:
          build_city_bonus = False
          build_city_bonus_off_reason = '(no path found)'
        else:
          arrival_turns = tmp_arrival_turns

      opponent_weight = 0
      if is_fuelable_near_resource_tile:
        debug = False
        if worker.id in DRAW_UNIT_LIST and near_resource_tile.pos in MAP_POS_LIST:
          debug = True

        # if worker.id in DRAW_UNIT_LIST and near_resource_tile.pos in MAP_POS_LIST:
          # print(
              # f' worker in unit = {worker.id in DRAW_UNIT_LIST}@{worker.id} pos in LIST = {near_resource_tile.pos in MAP_POS_LIST},   {near_resource_tile.pos}, debug={debug}'
          # )
        oppo_arrival_turns, nearest_oppo_unit = self.get_nearest_opponent_unit_to_cell(
            near_resource_tile, debug=debug)
        if oppo_arrival_turns < MAX_PATH_WEIGHT:
          opponent_weight += 1e-3 / dd(oppo_arrival_turns)

        # Use a large weight to defend my resource
        if (nearest_oppo_unit and
            oppo_arrival_turns <= MIN_DEFEND_ENEMY_ARRIVAL_TRUNS and
            arrival_turns <= oppo_arrival_turns):
          unit_nearest_cluster_ids = self.ci.get_opponent_unit_nearest_cluster_ids(
              nearest_oppo_unit, debug=debug)
          cell_cluster_ids = self.ci.get_neighbour_cells_cluster_ids(
              near_resource_tile)

          focus_cluster_ids = unit_nearest_cluster_ids & cell_cluster_ids
          if focus_cluster_ids:
            cell_is_nearest_in_cluster = any(
                (self.ci.
                 get_min_turns_to_cluster_near_resource_cell_for_opponent_unit(
                     cid, nearest_oppo_unit) == oppo_arrival_turns)
                for cid in focus_cluster_ids)
            if cell_is_nearest_in_cluster:
              boost = (10001 if arrival_turns < oppo_arrival_turns else 1001)
              opponent_weight += boost

            if worker.id in DRAW_UNIT_LIST and near_resource_tile.pos in MAP_POS_LIST:
              # prt(f"focus_cluster_ids={focus_cluster_ids}")
              prt(f"min_near_cluster={self.ci.get_min_turns_to_cluster_near_resource_cell_for_opponent_unit(list(cell_cluster_ids)[0], nearest_oppo_unit)}"
                 )
              prt(f"nearest_oppo_unit={nearest_oppo_unit.id}, {near_resource_tile.pos} min_oppo_unit_turns={min_turns}, player_unit_arrival_turns={arrival_turns}, "
                  f"is_nearset_cluster_to_unit={bool(unit_nearest_cluster_ids & cell_cluster_ids)} "
                  f"cell_is_nearest_in_cluster={cell_is_nearest_in_cluster}")

      # Do not boost cluster owner to build city on non-target cluster with player citytile
      # exceptions:
      # 1) transfer build: so owner don't need that much resource to goto next cluster
      # 2) build citytile when defending.
      if (worker.is_cluster_owner and not is_next_to_target_cluster and
          self.ci.cell_has_player_citytile_on_target_cluster(
              worker, near_resource_tile) and
          ((not is_transfer_build_position) and (opponent_weight < 1))):
        build_city_bonus = False
        build_city_bonus_off_reason = '(cluster_owner)'

      # Keep at least X near resource tile for a coal or urnaium cluster
      n_boundary, n_open = self.cluster_info.count_min_boundary_near_resource_tiles(
          near_resource_tile)

      # prt(f"cell[{near_resource_tile.pos}] {n_open}")
      if (n_open <= KEEP_RESOURCE_OPEN or
          near_resource_tile.pos in self.blacklist_city_building_positions or
          near_resource_tile.pos in self.non_wood_resource_locations):
        build_city_bonus = False
        build_city_bonus_off_reason = f'(open_cell={n_open}, blicklist={near_resource_tile.pos in self.blacklist_city_building_positions}), non_wood_res_loc={near_resource_tile.pos in self.non_wood_resource_locations}'

      # Too large the build city bonus will cause worker divergence from its coal mining position

      # TODO: why need this threshold?
      # if (build_city_bonus and arrival_turns <= 3):
      if build_city_bonus:
        wt += 1001

        # Encourage worker to build connected city tiles.
        if near_resource_tile.n_citytile_neighbour > 0:
          wt += near_resource_tile.n_citytile_neighbour / 10

        # mark build city cell
        self.worker_build_city_tasks.add((worker.id, near_resource_tile.pos))

      # if near_resource_tile.pos in MAP_POS_LIST:
      # prt(f'c[{near_resource_tile.pos}] @1, wt={wt}, arrival_turns={arrival_turns}')

      debug = False
      if worker.id in DRAW_UNIT_LIST and near_resource_tile.pos in MAP_POS_LIST and plan_idx == 0:
        debug = True

      # Try to hide next to resource grid.
      if is_resource_tile_can_save_dying_worker(near_resource_tile,
                                                worker,
                                                debug=debug):
        wt += UNIT_SAVED_BY_RES_WEIGHT

      demote_opponent_unit = 0
      # if self.has_can_act_opponent_unit_as_neighbour(near_resource_tile):
      # demote_opponent_unit = -0.1

      v = ((wt) / dd(arrival_turns) + boost_cluster + fuel_wt +
           opponent_weight + transfer_build_wt + demote_opponent_unit +
           default_res_wt)
      if debug:
        prt(f'w[{worker.id}] nrt[{near_resource_tile.pos}] @last, v={v}. wt={wt}, clustr={boost_cluster}, fuel_wt={fuel_wt}'
            f' collect_amt={amount} opponent={opponent_weight}, transfer_build_wt={transfer_build_wt} arr_turns={arrival_turns}, build_city={build_city_bonus}, off={build_city_bonus_off_reason}'
            f' is_transfer_build_position={is_transfer_build_position}, demoet_oppo_unit={demote_opponent_unit}'
            f' default_res_wt={default_res_wt}, n_open={n_open}, in_non_wood=({near_resource_tile.pos in self.non_wood_resource_locations})'
           )
        # prt(f' {self.worker_build_city_tasks}')
      return v

    def get_boost_transfer_build_weight(worker, cell):
      # If only do one transfer, then need to sort the positions
      wt = 0
      if cell.pos in worker.transfer_build_locations:
        nb_citytiles = [
            nb_cell.citytile
            for nb_cell in get_neighbour_positions(
                cell.pos, self.game_map, return_cell=True)
            if cell_has_player_citytile(nb_cell, self.game)
        ]
        # assert len(nb_citytiles) > 0, f"cell at {cell.pos} has not nb citytile, w={worker.id}"
        wt = CLUSTER_BOOST_WEIGHT * 2 * len(nb_citytiles)

      # Stick worker onto citytile neighoubr to build city tile
      offer = self.accepted_transfer_offers.get(worker.id)
      if offer:
        offer_turn, offer_pos = offer
        if (self.game.turn - offer_turn <= 2 and offer_pos == worker.pos and
            worker.pos == cell.pos):
          wt = MAX_WEIGHT_VALUE

      # if worker.id in DRAW_UNIT_LIST and cell.pos in MAP_POS_LIST and plan_idx == 0:
      # prt(f" {worker.id} t={self.game.turn}, transfer offer={offer}")
      return wt

    def get_city_tile_neighbour_weight(worker, cell):
      wt = -99999
      boost = get_boost_transfer_build_weight(worker, cell)
      if boost > 0:
        wt = boost

        # TODO: only build on connection position
        # Also build city tile on city tile neighbour
        self.worker_build_city_tasks.add((worker.id, cell.pos))
      return wt

    def get_worker_tile_weight(worker, target):
      wt = -99999
      if worker.pos == target.pos:
        wt = 1e-7
      return wt

    # Value matrix for ship target assginment
    # * row: workers
    # * column: point of interests
    target_cells = collect_target_cells(self.game.turn, self.game)
    weights = np.ones((len(workers), len(target_cells))) * -9999
    for i, worker in enumerate(workers):
      for j, target in enumerate(target_cells):
        # Can't arrive at target cell.
        quicker_path, quicker_dest_turns = self.select_quicker_path(
            worker, target.pos)
        if quicker_dest_turns >= MAX_PATH_WEIGHT:
          # if worker.id in DRAW_UNIT_LIST and target.pos in MAP_POS_LIST:
            # v = -9999
            # prt(f"CAN_NOT_ARRIVE: w[{worker.id}], cd={worker.cooldown}, t[{target.pos}], wt={v:.1f}", file=sys.stderr)
          continue

        # TODO: drop, because they might be using different time
        # Other worker can't move onto worker cell with cd > 1
        if target.unit and target.unit.cooldown >= 1 and worker.id != target.unit.id:
          continue

        if cell_has_player_citytile(target, self.game):
          v = get_city_tile_weight(worker, target, quicker_dest_turns)
          # if worker.id in DRAW_UNIT_LIST and target.pos in MAP_POS_LIST:
          # prt(f"to t[{target.pos}], v={v}, arr={quicker_dest_turns}", file=sys.stderr)
        elif target.has_resource():
          v = get_resource_weight(worker, target, quicker_dest_turns,
                                  quicker_path)
        elif target.is_near_resource:
          #TODO: use path_wo_cc when build tiles
          v = get_near_resource_tile_weight(worker, target, quicker_dest_turns,
                                            quicker_path)
        elif target.n_citytile_neighbour > 0:
          v = get_city_tile_neighbour_weight(worker, target)
        else:
          v = get_worker_tile_weight(worker, target)

        weights[i, j] = v

        if plan_idx == 0 and worker.id in DRAW_UNIT_LIST and DRAW_UNIT_TARGET_VALUE and v > 0:
          pos = target.pos
          a = annotate.text(pos.x, pos.y, f'{v:.2f}', fontsize=32)
          self.actions.append(a)

    idle_workers = []
    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    for worker_idx, target_idx in zip(rows, cols):
      worker = workers[worker_idx]
      wt = weights[worker_idx, target_idx]

      if wt < 1e-5 and worker.target_pos == worker.pos:
        idle_workers.append(worker)
        continue

      worker.target_score = weights[worker_idx, target_idx]

      target = target_cells[target_idx]
      worker.target = target
      worker.target_pos = target.pos

      if DRAW_UNIT_ACTION and plan_idx == 1:
        x = annotate.x(worker.pos.x, worker.pos.y)
        c = annotate.circle(target.pos.x, target.pos.y)
        line = annotate.line(worker.pos.x, worker.pos.y, target.pos.x,
                             target.pos.y)
        self.actions.extend([c, line])

    return idle_workers

  @functools.lru_cache(maxsize=1024, typed=False)
  def has_can_act_opponent_unit_as_neighbour(self, cell):
    for nb_cell in get_neighbour_positions(cell.pos,
                                           self.game_map,
                                           return_cell=True):
      if (nb_cell.unit and nb_cell.unit.team == self.game.opponent.team and
          nb_cell.unit.can_act()):
        return True
    return False

  def compute_worker_moves(self):
    g = self.game
    player = g.player
    workers = [
        unit for unit in player.units
        if unit.is_worker() and not unit.has_planned_action
    ]

    MAX_MOVE_WEIGHT = 99999

    def get_step_weight(worker, next_position, quick_path, next_step_positions):
      """Only workers next 5 positions will be computed here."""
      # if worker.pos == next_position:
      # If a worker can't move, it's a stone.
      # if not worker.can_act():
      # return 1000

      # If woker is building city.
      # if worker.is_building_city:
      # return 1000

      assert worker.target_pos is not None
      assert next_step_positions is not None

      next_cell = g.game_map.get_cell_by_pos(next_position)
      # Can't move on opponent citytile.
      if (next_cell.citytile is not None and
          next_cell.citytile.team == self.game.opponent.team):
        return -MAX_MOVE_WEIGHT

      # Do not move onto a transfer worker or a can not act unit.
      if (next_cell.unit and next_cell.unit.team == player.team and
          (next_cell.unit.is_transfer_worker or not next_cell.unit.can_act())):
        return -MAX_MOVE_WEIGHT

      # Do not force worker into city tile other than it's target.
      if (quick_path.not_leaving_citytile and next_cell.citytile and
          next_cell.pos != worker.target.pos and worker_total_fuel(worker) > 0):
        return -MAX_MOVE_WEIGHT

      v = 0
      # Stay at current position.
      if next_position == worker.pos:
        v += 1

      if (next_position == worker.target_pos or
          next_position in next_step_positions):
        v += (worker.target_score + 10)
        # v += 50

        # Priority all positions of th dying worker, let others make room for him.
        if worker.is_cargo_not_enough_for_nights:
          v += 1000

        target_cell = self.game_map.get_cell_by_pos(worker.target_pos)
        if target_cell.has_resource():
          v += 1

        # Try step on resource: the worker version is better, maybe because
        # other worker can use that.
        amount, fuel = get_cell_resource_values(next_cell, g.player)
        if fuel > 0:
          v += 1

        if next_position in worker.transfer_build_locations:
          v += 1000

        # Prevent collison with enemy when possible.
        if self.has_can_act_opponent_unit_as_neighbour(next_cell):
          v -= 0.1

      if worker.id in DRAW_UNIT_LIST and DRAW_UNIT_MOVE_VALUE:
        a = annotate.text(next_position.x,
                          next_position.y,
                          f'{int(v)}',
                          fontsize=32)
        self.actions.append(a)

      # if worker.id in DRAW_UNIT_LIST:
      # prt(f"w[{worker.id}]@{worker.pos}, next[{next_position}], v={v}, target={worker.target_pos}, next_points={path_positions}", file=sys.stderr)
      return v

    def gen_next_positions(worker):
      if not worker.can_act():
        return [worker.pos]

      # TODO: skip non-reachable positions?
      return [worker.pos] + [
          c.pos for c in get_neighbour_positions(
              worker.pos, g.game_map, return_cell=True)
      ]

    next_positions = {
        pos for worker in workers for pos in gen_next_positions(worker)
    }

    def duplicate_positions(positions):
      for pos in positions:
        cell = self.game.game_map.get_cell_by_pos(pos)
        if cell.citytile is not None and cell.citytile.team == self.game.id:
          for _ in range(MAX_UNIT_PER_CITY):
            yield pos
        else:
          yield pos

    next_positions = list(duplicate_positions(next_positions))

    # prt(f'turn={g.turn}, compute_worker_moves next_positions={len(next_positions)}',
    # file=sys.stderr)
    def get_position_to_index():
      d = defaultdict(list)
      for i, pos in enumerate(next_positions):
        d[pos].append(i)
      return d

    position_to_indices = get_position_to_index()
    C = np.ones((len(workers), len(next_positions))) * -MAX_MOVE_WEIGHT
    for worker_idx, worker in enumerate(workers):
      assert worker.target is not None

      quick_path, _ = self.select_quicker_path(worker, worker.target_pos)
      target_pos = worker.target.pos
      is_worker_deliver_resource = (cell_has_player_citytile(
          worker.target, self.game) and cargo_total_amount(worker.cargo) > 0)
      if (is_worker_deliver_resource or
          self.is_worker_building_citytile(worker, target_pos) or
          self.is_worker_fuel_city_task(worker, target_pos)):
        # Use path no passing citytile
        _, quick_path = self.quickest_path_pairs[worker.id]
        # if worker.id in DRAW_UNIT_LIST:
        # a, b = self.quickest_path_pairs[worker.id]
        # prt(f' a={a.not_leaving_citytile}, b={b.not_leaving_citytile}')

      # if worker.id in ['u_2']:
      # prt(f'-> {worker.id}, amt={worker.cargo}, building_tile: {is_worker_building_citytile(worker)}')
      # if is_worker_building_citytile(worker):
      # prt(f'not leave city{quick_path.not_leaving_citytile}, path_positions={quick_path.get_next_step_path_points(worker.target_pos)}')

      # path staths from dest to start point
      # if worker.target_pos is None:
      # prt(f'  w[{worker.id}]={worker.pos}, target={worker.target_pos}, S_path={path_positions}', file=sys.stderr)

      next_step_positions = quick_path.get_next_step_path_points(
          worker.target_pos, worker.pos)
      # if worker.id in DRAW_UNIT_LIST:
      # prt(f'next_step_positions: {next_step_positions}, is_building_city={self.is_worker_building_citytile(worker, target_pos)}, {self.worker_build_city_tasks}, target_pos={worker.target_pos}')
      self.actions.extend(quick_path.actions)

      for next_position in gen_next_positions(worker):
        wt = get_step_weight(worker, next_position, quick_path,
                             next_step_positions)
        C[worker_idx, position_to_indices[next_position]] = wt
        # if DEBUG:
        # prt(f'w[{worker.id}]@{worker.pos} goto {next_position}, wt = {wt}', file=sys.stderr)

    # prt(f'turn={g.turn}, compute_worker_moves before linear_sum_assignment',
    # file=sys.stderr)
    rows, cols = scipy.optimize.linear_sum_assignment(C, maximize=True)
    for worker_idx, poi_idx in zip(rows, cols):
      worker = workers[worker_idx]
      if not worker.can_act():
        continue

      wt = C[worker_idx, poi_idx]
      if wt < 0:
        prt((f"w[{worker.id}]@{worker.pos}, next[{next_position}], "
             f"v={C[worker_idx, poi_idx]}, target={worker.target_pos}"),
            file=sys.stderr)
        assert wt >= 0

      next_position = next_positions[poi_idx]
      next_cell = self.game_map.get_cell_by_pos(next_position)

      # Do not take any action at last night, unless go into citytile
      if (is_last_night(self.game.turn) and next_position != worker.pos and
          not cell_has_player_citytile(next_cell, self.game)):
        prt(f"ignore w[{worker.id}]@{worker.pos}=>[{next_position}]")
        continue

      move_dir = worker.pos.direction_to(next_position)
      self.add_unit_action(worker, worker.move(move_dir))

  @functools.lru_cache(maxsize=512)
  def has_only_one_wood_neighbour_resource_tile(self, cell):
    n_resource = 0
    res_type = None
    for nb_cell in get_neighbour_positions(cell.pos,
                                           self.game_map,
                                           return_cell=True):
      if is_resource_researched(nb_cell.resource, self.game.player):
        n_resource += 1
        res_type = nb_cell.resource.type
    return n_resource == 1 and res_type == Constants.RESOURCE_TYPES.WOOD

  def try_build_citytile(self, dry_run=False):
    t = self.game.turn % CIRCLE_LENGH
    city_building_units = []
    for unit in self.game.player.units:
      unit.is_building_city = False
      if not unit.is_worker():
        continue

      # Sitting on th cell of target position for city building.
      # Sit and build.
      if (self.circle_turn < BUILD_CITYTILE_ROUND and unit.can_act() and
          unit.can_build(self.game.game_map) and
          (unit.target_pos and unit.target_pos == unit.pos and
           self.is_worker_building_citytile(unit, unit.target_pos))):
        cell = self.game_map.get_cell_by_pos(unit.pos)

        # TODO: check neighbour city can survive.
        # Do not build single city on first night, or only one wood resoruce.
        if (is_first_night(self.game.turn) and
            cell.n_citytile_neighbour == 0 and
            self.has_only_one_wood_neighbour_resource_tile(cell)):
          continue

        city_building_units.append(unit)
        if dry_run:
          continue

        self.add_unit_action(unit, unit.build_city())
        unit.is_building_city = True
        if unit.cargo.wood == CITY_BUILD_COST:
          self.is_wood_city_tile.add(unit.pos)

        if unit.id in DRAW_UNIT_LIST:
          prt(f"BUILD_CITY: t={self.game.turn} u={unit.id}<{unit.cargo}>, target_pos={unit.target_pos}, is_build_city_task={self.is_worker_building_citytile(unit, unit.target_pos)}, tranfer_offer={self.accepted_transfer_offers.get(unit.id)}"
             )
    return city_building_units

  def compute_citytile_actions(self):
    player = self.game.player
    total_tile_count = player.city_tile_count
    total_unit_count = len(player.units)

    cur_research_points = player.research_points

    def get_city_action_weight(city):
      cluster_ids = set()
      for cell in city.citytiles:
        cluster_ids |= self.ci.get_neighbour_cells_cluster_ids(cell)

      cluster_ids = {
          cid for cid in cluster_ids if is_resource_researched(
              Resource(self.ci.get_cluster_type(cid), 1), self.game.player)
      }
      positions = set()
      for cid in cluster_ids:
        cluster = self.ci.c(cid)
        open_positions = cluster.get_open_boundary_positions(can_build=True)
        positions |= open_positions
      return len(positions), get_city_no(city)

    cities = list(player.cities.values())
    cities = sorted(cities,
                    key=lambda c: get_city_action_weight(c),
                    reverse=True)
    # for c in cities:
    # prt(f"{c.id} {get_city_action_weight(c)}")

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

        city = get_player_city_by_citytile(citytile, self.game)
        # print(f"citytile={citytile.pos}, from city={city.id}, build_worker()")

    cities = sorted(cities, key=lambda c: get_city_action_weight(c))
    for citytile in every_citytiles(cities):
      if citytile.pos in action_citytile_positions:
        continue
      if cur_research_points < URANIUM_RESEARCH_POINTS:
        cur_research_points += 1
        self.actions.append(citytile.research())

  def update_player_info(self):
    # Estimate number of research point in left day times.
    # n_city = len(self.player.cities)
    self.player.avg_research_point_growth = self.player.city_tile_count / CITY_ACTION_COOLDOWN

    opponent = self.game.opponent
    opponent.avg_research_point_growth = opponent.city_tile_count / CITY_ACTION_COOLDOWN

  def update_city_info(self):
    remaining_nights = get_remaining_nights(self.game.turn)
    nights_to_next_day = get_night_count_this_round(self.game.turn)
    for city in self.game.player.cities.values():
      city.last_turns = city_last_days(self.game.turn, city)
      city.city_game_end_fuel_req = (remaining_nights * city.light_upkeep -
                                     city.fuel)
      city.city_next_day_fuel_req = (nights_to_next_day * city.light_upkeep -
                                     city.fuel)
      city.is_dying_this_round = (city.city_next_day_fuel_req > 0)
      city.later_round_nights_to_live = remaining_nights - nights_to_next_day

  @property
  def ci(self):
    return self.cluster_info

  @timeit
  def update_game_info(self):
    self.cluster_info = ClusterInfo(self.game)
    self.cluster_info.cluster()

    self.update_player_info()
    self.update_game_map_info()

    self.update_unit_info()
    self.update_city_info()

  def assign_worker_to_resource_cluster(self, multi_worker=False):
    """For each no citytile cluster of size >= 2, find a worker with cargo space not 100.
      And set its target pos."""
    # Computes worker property.
    # if self.game.player.city_tile_count < 2:
      # return

    if self.cluster_info.max_cluster_id == 0:
      return

    MIN_CLUSTER_WT = -0.001

    def get_cluster_weight(worker, cluster):
      # TODO: MAYBE keep it?
      if worker.target_city_id:
        return MIN_CLUSTER_WT

      debug = False
      if worker and worker.id in DRAW_UNIT_LIST:
        debug = True

      cid = cluster.cid
      fuel = cluster.total_fuel
      arrival_turns, tile_positions = cluster.worker_cluster_turns_info(worker)

      # Save tile position
      tile_pos = (tile_positions and tile_positions[0] or None)
      worker.cid_to_tile_pos[cid] = tile_pos  # could be near resource tile
      worker.cid_to_cluster_turns[cid] = arrival_turns
      if arrival_turns >= MAX_PATH_WEIGHT:
        return MIN_CLUSTER_WT

      # resource cluster not researched.
      cell = self.game_map.get_cell_by_pos(tile_pos)

      quick_path, _ = self.select_quicker_path(worker, tile_pos)
      surviving_turns = get_surviving_turns_at_cell(worker, quick_path, cell)

      cluster_type = cluster.resource_type
      wait_turns = resource_researched_wait_turns(
          Resource(cluster_type, 1),
          self.player,
          move_days=arrival_turns,
          surviving_turns=surviving_turns)

      # if debug:
      # prt(f"send worker{worker.id} to cluster{tile_pos}, survive={surviving_turns}, wait_turns={wait_turns}")
      # TODO: should i limit it?
      MAX_WAIT_ON_CLUSTER_TURNS = 8
      if wait_turns < 0 or wait_turns > MAX_WAIT_ON_CLUSTER_TURNS:
        return MIN_CLUSTER_WT

      boundary_positions = cluster.boundary_positions
      open_positions = cluster.get_open_boundary_positions()
      open_ratio = len(open_positions) / len(boundary_positions)

      # dying_boost = 0
      # if worker.is_cargo_not_enough_for_nights:
      # dying_boost = 2
      wt = fuel * open_ratio / dd((arrival_turns + wait_turns), r=1.1)
      # prt(f"t={self.game.turn}, edge {worker.id}, c@{tile_pos} fuel={fuel}, wait={wait_turns}, arrival_turns={arrival_turns}, wt={wt}, open_ratio={open_ratio}")
      return wt

    def gen_resource_clusters():
      for c in self.cluster_info.clusters.values():
        # TODO(wangfei): remove this
        # n_resource_tile = c.size
        # if (n_resource_tile <= 1 and
            # c.resource_type == Constants.RESOURCE_TYPES.WOOD):
          # continue
        yield c

    # RESOURCE_WORKER_RATIO = 3
    SPLIT_BOUNDARY_CNT = 15

    def gen_multi_worker_resource_clusters():
      for c in self.cluster_info.clusters.values():
        yield c

        n_open_boundary_size = len(c.get_open_boundary_positions())
        # Add extra worker
        if n_open_boundary_size > SPLIT_BOUNDARY_CNT:
          yield c
        # n_resource_tile = c.size
        # n_workers = int(math.ceil(n_resource_tile / RESOURCE_WORKER_RATIO))
        # if n_workers > 0:
          # for _ in range(n_workers):
            # yield c

    workers = self.player_available_workers()
    if multi_worker:
      workers = [w for w in workers if w.target_cluster_id < 0]

    resource_clusters = (list(gen_resource_clusters()) if not multi_worker else
                         list(gen_multi_worker_resource_clusters()))
    print(f'Number of cluster: {len(resource_clusters)}')

    weights = np.ones((len(workers), len(resource_clusters))) * MIN_CLUSTER_WT
    for j, cluster in enumerate(resource_clusters):
      for i, worker in enumerate(workers):
        weights[i, j] = get_cluster_weight(worker, cluster)

    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    sorted_pairs = sorted(list(zip(rows, cols)),
                          key=lambda x: -weights[x[0], x[1]])
    for worker_idx, cluster_idx in sorted_pairs:
      worker = workers[worker_idx]
      cluster = resource_clusters[cluster_idx]

      # tile_pos = worker.cid_to_tile_pos[cid]
      # prt(f'Assign Cluster {worker.id}, cell[{tile_pos}], wt={weights[worker_idx, cluster_idx]}')
      if weights[worker_idx, cluster_idx] < 0:
        continue

      # Keep the mapping, but not do cluster boost (not good)
      # n_player_citytile = self.ci.c(cid).player_citytile_count
      # if n_player_citytile > 0:
      # continue

      cid = cluster.cid
      worker.target_cluster_id = cid
      worker.target_cluster = cluster
      worker.target_cid_turns = worker.cid_to_cluster_turns[cid]
      worker.is_cluster_owner = True

      if DRAW_UNIT_CLUSTER_PAIR:
        tile_pos = worker.cid_to_tile_pos[cid]
        x = annotate.x(tile_pos.x, tile_pos.y)
        line = annotate.line(worker.pos.x, worker.pos.y, tile_pos.x, tile_pos.y)
        self.actions.extend([x, line])
        # prt(f'Assign Cluster {worker.id}, cell[{tile_pos}], wt={weights[worker_idx, cluster_idx]}')

  def worker_look_for_resource_transfer(self):
    circle_turn = self.circle_turn
    if circle_turn >= BUILD_CITYTILE_ROUND:
      return

    def cell_can_build_citytile(cell, sender_unit=None):
      if sender_unit and cell.pos == sender_unit.pos:
        return False

      is_resource_tile = cell.has_resource()
      is_citytile = cell.citytile is not None
      unit_can_not_act = (cell.unit and not cell.unit.can_act())
      return (not is_resource_tile and not is_citytile and not unit_can_not_act)

    def unit_can_collect_and_build(cell, unit):
      cell_can_build = cell_can_build_citytile(cell, unit)
      if not cell_can_build:
        return False

      # Build directly.
      unit_amt = worker_total_cargo(unit)
      if unit_amt >= CITY_BUILD_COST:
        return True

      # Collect then build
      collect_amt, _ = get_one_step_collection_values(cell,
                                                      self.game.player,
                                                      self.game,
                                                      unit=unit)
      return unit_amt + collect_amt >= CITY_BUILD_COST

    def neighbour_cells_can_collect_and_build_citytile(cell, sender_unit):
      for nb_cell in get_neighbour_positions(cell.pos,
                                             self.game_map,
                                             return_cell=True):
        if unit_can_collect_and_build(nb_cell, sender_unit):
          return True
      return False

    def get_neighbour_cell_weight(cell):
      wt = 0
      for nb_cell in get_neighbour_positions(cell.pos,
                                             self.game_map,
                                             return_cell=True):
        if cell_has_player_citytile(nb_cell, self.game):
          wt += 100
        elif is_resource_researched(nb_cell.resource, self.player):
          wt += 1000

        # Boost position of opponent near cells.
        min_turns, min_oppo_unit = self.get_nearest_opponent_unit_to_cell(cell)
        if min_oppo_unit:
          wt += 1 / (min_turns or 1)
      return wt

    def neighbour_worker_with_enough_resource(worker):
      """Assign transfer action here, but leave movement to the existing process."""
      worker_amt = worker_total_cargo(worker)
      nb_cells = get_neighbour_positions(worker.pos,
                                         self.game_map,
                                         return_cell=True)
      next_cells = nb_cells + [self.game_map.get_cell_by_pos(worker.pos)]

      # Sort to prioritize better cells.
      next_cells = sorted(next_cells,
                          key=lambda c: -get_neighbour_cell_weight(c))
      for target_cell in next_cells:
        # if worker.id in DRAW_UNIT_LIST:
        # prt(f" c={target_cell.pos}, wt={get_neighbour_cell_weight(target_cell)}")
        # Target cell must can build citytile
        if not cell_can_build_citytile(target_cell):
          continue

        # TODO: try make one step collections value more accurate
        # Worker can collect on its own.
        collect_amt = collect_amount_at_cell(target_cell, self.player,
                                             self.game_map)
        if worker_amt + collect_amt >= CITY_BUILD_COST:
          break

        # transfer + worker.action + worker.collect on next step
        for nb_cell in nb_cells:
          nb_unit = nb_cell.unit
          debug = False
          # if nb_unit and nb_unit.id == 'u_4' and worker.id == 'u_6':
          # debug = True
          # prt(f" nb_unit {nb_unit.id}, left={nb_unit.get_cargo_space_left()} has_planned_action={nb_unit.has_planned_action}, can_act={nb_unit.can_act()}")

          # Neighbour is not valid or ready.
          if (nb_unit is None or nb_unit.team != self.game.player.team or
              not nb_unit.can_act() or nb_unit.has_planned_action):
            continue

          # Neighbour can build on its own cell.
          is_nb_unit_can_build = cell_can_build_citytile(nb_cell)
          is_nb_unit_can_collect_and_build = (
              neighbour_cells_can_collect_and_build_citytile(nb_cell, nb_unit))
          if is_nb_unit_can_build or is_nb_unit_can_collect_and_build:
            continue

          # TODO: support other resource: use max resource
          if (worker_amt + collect_amt + nb_unit.cargo.wood >= CITY_BUILD_COST):
            transfer_amount = CITY_BUILD_COST - (worker_amt + collect_amt)
            prt(f'$A {worker.id}{worker.cargo}@{worker.pos} accept transfer from {nb_unit.id}{nb_unit.cargo} ${transfer_amount} to goto {target_cell.pos}',
                file=sys.stderr)
            self.add_unit_action(
                nb_unit,
                nb_unit.transfer(worker.id, RESOURCE_TYPES.WOOD,
                                 transfer_amount))
            nb_unit.is_transfer_worker = True
            worker.transfer_build_locations.add(target_cell.pos)

            # TODO: maybe add worker action directly here?
            self.accepted_transfer_offers[worker.id] = (self.game.turn,
                                                        target_cell.pos)

            # Assume the first worker transfer.
            return

    # check direct transfer and build.
    # Pre-condition: worker.can_act & unit.can_act
    for worker in self.game.player.units:
      if worker.has_planned_action:
        continue
      if not worker.is_worker() or not worker.can_act():
        continue

      # Do not tranfer for worker on resource
      if is_resource_researched(worker.cell.resource, self.game.player):
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
      for nb_cell in get_neighbour_positions(worker.pos,
                                             self.game_map,
                                             return_cell=True):
        if cell_has_player_citytile(nb_cell, self.game):
          yield nb_cell.citytile.cityid

    MIN_CITY_WEIGHT = -9999

    def get_city_weight(worker, city):
      # If current worker won't help, skip it.
      if not worker.can_act():
        return MIN_CITY_WEIGHT

      # There is a cell for worker to go back and save itself.
      # _, worker_collect_fuel = get_cell_resource_values(worker.cell, self.player)
      # is_safe_cell = (worker_collect_fuel >= get_unit_upkeep(worker))
      # if not is_safe_cell:
      # return MIN_CITY_WEIGHT

      # Do not go to city if city can wait and worker is not full enough.
      city_left_days = math.floor(city.fuel / city.light_upkeep)
      worker_cargo_amt = worker_total_cargo(worker)
      if city_left_days > 0 and worker_cargo_amt < 95:
        return MIN_CITY_WEIGHT

      worker_fuel = worker_total_fuel(worker)
      city_left_days_deposited = math.floor(
          (city.fuel + worker_fuel) / city.light_upkeep)
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
      # prt(f'{worker.id} is sending to {city.cityid}')

  def is_worker_building_citytile(self, worker, target_pos):
    return (worker.id, target_pos) in self.worker_build_city_tasks

  def is_worker_fuel_city_task(self, worker, target_pos):
    return (worker.id, target_pos) in self.worker_fuel_city_tasks

  def send_worker_to_target(self):

    def clear_idle_worker_tasks(tasks, worker_ids):
      remove_tasks = {(uid, pos) for uid, pos in tasks if uid in worker_ids}
      return tasks - remove_tasks

    self.worker_build_city_tasks = set()
    self.worker_fuel_city_tasks = set()
    workers = self.player_available_workers()
    self.assign_worker_target(workers, plan_idx=0)

    # Second plan to remove over fuel city worker to other tasks
    self.fuel_city_by_transfer_positions = {}
    self.keep_city_alive_till_game_end = set()  # (unit.id, city.id)
    self.keep_city_alive_till_next_day = set()  # (unit.id, city.id)
    self.transfer_resource_to_city_tile()

    self.blacklist_city_building_positions = set()
    self.check_cluster_closed_boundary()

    self.worker_build_city_tasks = set()
    self.worker_fuel_city_tasks = set()
    workers = self.player_available_workers(
    )  # worker may do tranfer and become no avialable.

    self.idle_woker_ids = set()
    replan_workers = self.assign_worker_target(workers, plan_idx=1)
    for u in replan_workers:
      u.is_idle_worker = True
      self.idle_woker_ids.add(u.id)

    # TODO: Is it needed?
    worker_ids = {w.id for w in replan_workers}
    self.worker_build_city_tasks = clear_idle_worker_tasks(
        self.worker_build_city_tasks, worker_ids)
    self.worker_fuel_city_tasks = clear_idle_worker_tasks(
        self.worker_fuel_city_tasks, worker_ids)

    idle_workers2 = set()
    if replan_workers:
      idle_workers2 = self.assign_worker_target(replan_workers, plan_idx=2)
    prt(f"I1={len(replan_workers)}, I2={len(idle_workers2)}", file=sys.stderr)

  def check_cluster_closed_boundary(self):
    """Check build city position will block resource going out of cluster."""
    city_building_units = self.try_build_citytile(dry_run=True)
    city_building_positions = {unit.pos for unit in city_building_units}

    cluster_ids = set()
    for unit in city_building_units:
      cids = self.ci.get_neighbour_cells_cluster_ids(unit.target)
      cluster_ids |= cids

    for cluster in self.ci.get_clusters(cluster_ids):
      # Only check coal and uranium cluster.
      if cluster.resource_type == Constants.RESOURCE_TYPES.WOOD:
        continue

      open_positions = cluster.get_open_boundary_positions()
      build_positions = open_positions & city_building_positions
      if len(open_positions) - len(build_positions) < KEEP_RESOURCE_OPEN:
        max_build = max(len(open_positions) - KEEP_RESOURCE_OPEN, 0)
        n = len(build_positions) - max_build

        for pos in list(build_positions)[:n]:
          self.blacklist_city_building_positions.add(pos)

  def transfer_resource_to_city_tile(self):
    """Use transfer resoruce to city if:
    0) worker assigned fuel city task and can_act() and next to target
    1) the target pos: is city tile and has player unit
    2) Unit cargo contains fuel more than the city required to the end of game.
    """

    def select_unit_transfer_resource(unit):
      cargo = unit.cargo
      resources = [(WOOD_FUEL_RATE, cargo.wood, RESOURCE_TYPES.WOOD),
                   (COAL_FUEL_RATE, cargo.coal, RESOURCE_TYPES.COAL),
                   (URANIUM_FUEL_RATE, cargo.uranium, RESOURCE_TYPES.URANIUM)]
      resources.sort(key=lambda x: x[0] * x[1], reverse=True)
      return resources[0]

    def collect_city_rescue_workers():
      for unit in self.game.player.units:
        # Discard not fuel city task unit.
        target_pos = unit.target.pos
        if not self.is_worker_fuel_city_task(unit, target_pos):
          continue

        _, quick_path = self.quickest_path_pairs[unit.id]
        arrival_turns = quick_path.query_dest_turns(target_pos)
        yield arrival_turns, unit

    def get_fuel_req(city):
      if city.is_dying_this_round:
        return city.city_next_day_fuel_req
      return city.city_game_end_fuel_req

    def add_fuel_unit(unit, city):
      unit_fuel = cargo_total_fuel(unit.cargo)

      if city.is_dying_this_round:
        self.keep_city_alive_till_next_day.add((unit.id, city.id))
        fuel_req = city.city_next_day_fuel_req
        after_fuel_req = fuel_req - unit_fuel
        city.city_next_day_fuel_req = after_fuel_req
      else:
        # save it till the end
        self.keep_city_alive_till_game_end.add((unit.id, city.id))
        fuel_req = city.city_game_end_fuel_req
        after_fuel_req = fuel_req - unit_fuel
        city.city_game_end_fuel_req = after_fuel_req

      # If there is not much overhead fuel from the unit, just move onto citytile.
      later_nights_fuel = 0
      if city.is_dying_this_round:
        later_nights_fuel = city.later_round_nights_to_live * city.light_upkeep
      if unit_fuel - fuel_req - later_nights_fuel < 40:
        # this is move to fuel
        prt(f"fuel city={city.id} by w={unit.id}@{unit.pos}, dying_this_round={city.is_dying_this_round}, move_fuel={unit_fuel}, before={fuel_req}, after={after_fuel_req}"
           )
        return -1

      # Use transfer to save city to the end
      return fuel_req

    def transfer_fuel(unit, city, fuel_req):
      fuel_rate, amt, res_type = select_unit_transfer_resource(unit)

      req_amt = int(math.ceil(fuel_req / fuel_rate))
      transfer_amt = min(amt, req_amt)

      unit_fuel = cargo_total_fuel(unit.cargo)
      self.fuel_city_by_transfer_positions[unit.target.pos] = unit_fuel

      after_fuel = fuel_req - transfer_amt * fuel_rate
      prt(f"fuel city={city.id} by w={unit.id}@{unit.pos}, dying_on_rond={city.is_dying_this_round}, transfer_fuel={unit_fuel}, before={fuel_req}, after_tranfer={after_fuel}"
         )

      if not unit.can_act():
        return

      # Not in the right position to do transfer.
      if unit.pos.distance_to(unit.target.pos) != 1:
        return

      # No receptor live in city.
      city_unit = unit.target.unit
      if city_unit is None:
        # Unit **waits** for city unit to receive fuel.
        if city.last_turns > 0:
          move_dir = unit.pos.direction_to(unit.pos)
          self.add_unit_action(unit, unit.move(move_dir))
        return

      prt(f'$B {unit.id}{unit.cargo}@{unit.pos} transfer {transfer_amt} to city {unit.target.pos}',
          file=sys.stderr)
      self.add_unit_action(unit,
                           unit.transfer(city_unit.id, res_type, transfer_amt))

    def try_save_city(unit, city):
      # City has enough resource till the end.
      if get_fuel_req(city) <= 0:
        return

      fuel_req = add_fuel_unit(unit, city)
      if fuel_req < 0:
        return

      # We do transfer delivery here!
      # Will add weight for this citytile target during replan to call for city onwer.
      transfer_fuel(unit, city, fuel_req)

    # Sort worker by their arrival_turns to city target
    rescue_units = list(collect_city_rescue_workers())
    rescue_units.sort(key=lambda x: (x[0], x[1].id))
    for _dist, unit in rescue_units:
      assert unit.target.citytile
      citytile = unit.target.citytile
      city = get_player_city_by_citytile(citytile, self.game)
      try_save_city(unit, city)

  @timeit
  def execute(self):
    actions = self.actions
    g = self.game
    player = g.player

    # MAIN_PRINT
    prt((
        f'>>T turn={g.turn}, #W={len(player.units)}, #C={player.city_tile_count} '
        f'R={player.research_points}'),
        file=sys.stderr)

    self.update_game_info()

    self.compute_citytile_actions()

    # self.assign_worker_city_target()
    self.worker_look_for_resource_transfer()

    self.assign_worker_to_resource_cluster()
    # self.assign_worker_to_resource_cluster(multi_worker=True)

    self.send_worker_to_target()

    self.try_build_citytile()

    self.compute_worker_moves()


_strategy = Strategy()


def agent(observation, configuration):
  _strategy.update(observation, configuration)
  _strategy.execute()
  return _strategy.actions
