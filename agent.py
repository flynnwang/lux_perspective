"""
- [√] remove limit (or limit at 120)
- [√] 860693002@t10: try build citytile next to each other.

(explore cluster optimization)
- [√] 242859934: also visit enemy's cluster for city tile building
- [√] 242859934: why the cluster explorer coming back and forworth?
      (cause resource weight is larger, use unit cargo as hint)

- [-] cluster boost with weight decay (so that it will persist)
  * This is already the case.
- [√ ]969408840@0t32u3: Do not move to coal too early
  * limit on wait_turns <= 6
  * count as arrival for near_resource_tile
  * t39: worker dying => fix sim_on_cell sim turn ref.
- [√] 969408840: do not use cluster boost worker to resurce cityhouse when it's wood full

- [√] 699414615/a1/t1: Bias cells towards enemy locations
  - [√] 699414615@t70u27: not moving back city (bug of bias towards enemy)




- Boost cluster explore task over city build task?
- collect edge count info for cluster assignment.
- * support wait on days for priority search (for resource research points)?

- add randomized actions
- ** Defend my city

- 242859934@t20 Try to resolve global & local conficts.


- [X] 796001604@t28: move out of dying citytile (not possible, no resoruce tile or other city.)


- Try defend enemy come into my cluster

- save size one coal city: not cheap to build
- **Ending pose: build extra city tile with transfer

- boost agent into city building position at next day - 4
  * remove build city turn limit, so worker will goto near resource tile natually

@668323264
- [TODO] t=41, u_11, goto remote tile without crossing city
 * u_2 t=20
- [x] goto build city tile without crossing city
  already did that

- [TODO] Should I boost more for near resource tile (for defence?) than resource tile.
- [TODO] boost worker into resource tile when possible?

- [TODO] wait in the day? for faster arrival.


@537926055
- why u_4 return to cluster 0?

- search based on enemy city light keep
- why u_5 goto coal far away at turn 28?
- wait too long at urnaium


@178969124 explore: invaded.




-> priority

- worker resource assignment, not very good (use global weight for boosting?), waiting but not on resouorce.
(this maybe a result of multiple worker to coal)

* Step out at at last night - 3, to build city tile on first day
- boost weight for neighbour city tile building (encourage citytile cluster)



- Build city tile on long sides (to connect)
* Limit the number of citytile per cluster


->
* Do not spawn at night in dying city.
* explore size = 1 wood
* Save size 1 citytile
* Tune BUILD_CITYTILE_ROUND to 30 (for the last day, must build connect city)
* worker forward fuel (from coal and uranium) to city better?

* Move earlier into wood for city building. (at first hour of day?)
* Support other type of resource transfer.

* give multiple worker to coal & uranium cell?
* [minor opt]: do not move to cell has limit resource due to collection: predict cell dying


* Use tranfer to save worker




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
from utility import *

DEBUG = True


DRAW_UNIT_ACTION = 1
DRAW_UNIT_CLUSTER_PAIR = 1


DRAW_UNIT_LIST = []
MAP_POS_LIST = [(15, 0)]
MAP_POS_LIST = [Position(x, y) for x, y in MAP_POS_LIST]
DRAW_UNIT_TARGET_VALUE = 0
DRAW_UNIT_MOVE_VALUE = 0
DRAW_QUICK_PATH_VALUE = 0



# TODO: add more
BUILD_CITYTILE_ROUND = CIRCLE_LENGH

MAX_PATH_WEIGHT = 99999

MAX_UNIT_NUM = 120


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


def is_resource_researched(resource, player, move_days=0, surviving_turns=0, debug=False):
  if resource is None:
    return False
  plus = max(move_days, surviving_turns) * player.avg_research_point_growth
  if debug:
    print(f' researched_coal={player.researched_coal(plus)}, '
          f'dR={player.avg_research_point_growth}, move_days={move_days},'
          f' surviving_turns={surviving_turns}, plus={plus}')
  if (resource.type == Constants.RESOURCE_TYPES.COAL
      and not player.researched_coal(plus)):
    return False
  if (resource.type == Constants.RESOURCE_TYPES.URANIUM
      and not player.researched_uranium(plus)):
    return False
  return True

def resource_researched_wait_turns(resource, player, move_days,
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
    if move_days + wait_turns > surviving_turns:
      return -1
    return wait_turns

  if resource.type == Constants.RESOURCE_TYPES.COAL:
    return waiting_turns(COAL_RESEARCH_POINTS)
  if resource.type == Constants.RESOURCE_TYPES.URANIUM:
    return waiting_turns(URANIUM_RESEARCH_POINTS)

  assert False, f"resource type not found: {resource.type}"
  return 0


def get_cell_resource_values(cell, player, unit=None, move_days=0,
                             surviving_turns=0):
  # Returns: (amount, fuel_weight)
  if not cell.has_resource():
    return 0, 0

  resource = cell.resource
  debug= False
  if unit and unit.id in DRAW_UNIT_LIST and cell.pos in MAP_POS_LIST:
    debug = True
  wait_turns = resource_researched_wait_turns(resource, player, move_days,
                                              surviving_turns, debug=debug)
  if wait_turns < 0:
    return 0, 0

  amount = get_worker_collection_rate(resource)
  amount = min(amount, resource.amount)
  if unit:
    amount = min(amount, unit.get_cargo_space_left())
  fuel = amount * get_resource_to_fuel_rate(resource)
  return amount, fuel / (move_days + wait_turns + 1)

# TODO(wangfei): try more accurate estimate
def resource_cell_added_surviving_nights(cell, upkeep, game):
  turns = resource_surviving_nights(game.turn, cell.resource, upkeep)
  for nb_cell in get_neighbour_positions(cell.pos, game.game_map,
                                         return_cell=True):
    turns += resource_surviving_nights(game.turn, nb_cell.resource, upkeep)
  return turns


def get_one_step_collection_values(cell, player, game, move_days=0, surviving_turns=0,
                                   unit=None):
  game_map = game.map
  amount, fuel_wt = get_cell_resource_values(cell, player, move_days=move_days,
                                          surviving_turns=surviving_turns,
                                          unit=unit)
  for nb_cell in get_neighbour_positions(cell.pos, game_map, return_cell=True):
    a, f = get_cell_resource_values(nb_cell, player, move_days=move_days,
                                    surviving_turns=surviving_turns,
                                    unit=unit)
    amount += a
    fuel_wt += f
  return amount, fuel_wt


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
        # Can not go pass through enemy citytile.
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


class SearchState:

  def __init__(self, turn, cell, cargo, cooldown):
    # print('SearchState ', type(turn), cell.pos, type(cooldown), cooldown)
    self.turn = turn
    self.cell = cell
    self.cargo = cargo
    self.cooldown = cooldown
    self.prev_positions = []
    self.deleted = False
    self.arrival_fuel = 0

  @property
  def pos(self):
    return self.cell.pos

  @property
  def fuel(self):
    return cargo_total_fuel(self.cargo)

  # @property
  # def key(self):
    # return (self.turn, self.pos.x, self.pos.y)

  def __lt__(self, other):
    return self.turn < other.turn or (self.turn == other.turn
                                      and self.fuel > other.fuel)

  def __eq__(self, other):
    return self.turn == other.turn and self.fuel == other.fuel


def sim_on_cell(turn, cargo, unit_type, next_cell, game,
                player, sim_turns, debug=False):
  """Sim one turn for worker move onto some cell, return cargo.

  1) collect resource
  2) drop resource to city.
  3) if night, make Units consume resources and CityTiles consume fuel
    - if unit no resource, removed
    - if city gone, unit gone with it
  """
  cargo = Cargo(cargo.wood, cargo.coal, cargo.uranium)

  def request_amount(cargo, res_type, n_res_type):
    if n_res_type == 0:
      return 0

    total = cargo_total_amount(cargo)
    if total == WORKER_RESOURCE_CAPACITY:
      return 0

    left_amount = WORKER_RESOURCE_CAPACITY - total
    amt = min(int(math.ceil(left_amount / n_res_type)),
              WORKER_COLLECTION_RATE[res_type])
    return amt

  def collect_resource(cargo):
    # TODO: may die because of other workers
    # For each type of resource, collect resource (assume no other wokers)
    for res_type in ALL_RESOURCE_TYPES:
      res_cells = res_type_to_cells[res_type]
      req_amt = request_amount(cargo, res_type, len(res_cells))
      if req_amt == 0:
        continue

      collect_amt = 0
      for c in res_cells:
        collect_amt += min(c.resource.amount, req_amt)
      add_resource_to_cargo(cargo, collect_amt, res_type)

  unit_upkeep = get_unit_upkeep_by_type(unit_type)
  def consume_cargo_resource(resource_amt, fuel_rate, request_amt):
    if request_amt == 0:
      return resource_amt, 0

    one_night_amount = int(math.ceil(unit_upkeep / fuel_rate))
    if resource_amt >= one_night_amount:
      return resource_amt - one_night_amount, 0
    return 0, request_amt - resource_amt * fuel_rate

  def consume_worker_resource(cargo, debug=False):
    request_amt = unit_upkeep
    # if debug:
      # print(f' r0={request_amt}')
    cargo.wood, request_amt = consume_cargo_resource(cargo.wood, WOOD_FUEL_RATE, request_amt)
    # if debug:
      # print(f' r1={request_amt}')
    cargo.coal, request_amt = consume_cargo_resource(cargo.coal, COAL_FUEL_RATE, request_amt)
    # if debug:
      # print(f' r2={request_amt}')
    cargo.uranium, request_amt = consume_cargo_resource(cargo.uranium, URANIUM_FUEL_RATE, request_amt)
    # if debug:
      # print(f' r3={request_amt}')
    return request_amt <= 0

  # count the number of neighbour resource types (assume cell not change)
  collect_cells = (get_neighbour_positions(next_cell.pos, game.game_map, return_cell=True)
                   + [next_cell])
  res_type_to_cells = defaultdict(list)
  for c in collect_cells:
    # TODO: Should i use estimate?
    if c.resource and is_resource_researched(c.resource, player):
      res_type_to_cells[c.resource.type.upper()].append(c)

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
  for t in range(turn, turn+sim_turns):
    collect_resource(cargo)

    # 2) drop resource onto city.
    if unit_on_citytile:
      city_fuel += cargo_total_fuel(cargo)
      cargo.clear()

    if debug:
      print(f't = {t}, is_night={is_night(t)}')
    if is_night(t):
      consumed = consume_worker_resource(cargo, debug=debug)
      if debug:
        print(f't = {t}, unit_on_citytile={unit_on_citytile}, consumed={consumed}')
      # unit die on non-city tile after resource consumption.
      if (not unit_on_citytile) and not consumed:
          return None

      # TODO: this is buggy, because city might already crash.
      # unit die on citytile at night.
      if unit_on_citytile:
        city_fuel -= city.light_upkeep
        if city_fuel < 0:
          return None
  return cargo


class QuickestPath:
  """Find the path with (min_turns, max_cargo_resource)."""

  MAX_FUTURE_TURNS = CIRCLE_LENGH

  def __init__(self, game, worker, not_leaving_citytile=False, debug=False):
    self.game = game
    self.turn = game.turn
    self.game_map = game.game_map
    self.worker = worker
    self.not_leaving_citytile = not_leaving_citytile
    self.state_map = [[None for h in range(self.game_map.height)]
                      for w in range(self.game_map.width)]
    self.debug = debug
    self.actions = []

  @property
  def max_turns(self):
   return min(self.turn + self.MAX_FUTURE_TURNS, MAX_DAYS)

  @property
  def worker_upkeep(self):
    return get_unit_upkeep(self.worker)

  def wait_for_cooldown(self, state, extra_wait_days=0):
    sim_turns = int(state.cooldown + extra_wait_days)
    cargo = sim_on_cell(state.turn, state.cargo, self.worker.type,
                        state.cell, self.game, self.player, sim_turns)
    if cargo is None:
      return None

    next_state = SearchState(state.turn + sim_turns, state.cell, cargo,
                             state.cooldown + extra_wait_days - sim_turns)
    next_state.arrival_fuel = cargo_total_fuel(state.cargo)
    return next_state

  def move(self, state, next_cell):
    cooldown = get_unit_action_cooldown(self.worker)
    if is_night(state.turn+1):
      cooldown *= 2
    is_citytile = cell_has_player_citytile(next_cell, self.game)
    if is_citytile:
      cooldown = 1  # 1 - 1 to get 0

    # Look ahead for cooldown to see whether worker will die on next move.
    cargo = sim_on_cell(state.turn, state.cargo, self.worker.type,
                        next_cell, self.game, self.player, sim_turns=cooldown)
    if cargo is None:
      return None

    debug = False
    if self.worker.id in DRAW_UNIT_LIST and self.turn == 39 and next_cell.pos == Position(11, 5):
      debug = True
    cargo = sim_on_cell(state.turn, state.cargo, self.worker.type,
                        next_cell, self.game, self.player, sim_turns=1, debug=debug)
    if debug:
      print(f' turn={state.turn} cargo_0: {state.cargo}, cargo_1: {cargo}, next={next_cell.pos}')
    if cargo is None:
      return None

    next_state = SearchState(state.turn+1, next_cell, cargo, cooldown-1)
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
          # print(f' push_head: prev{prev_state and prev_state.pos} to new{new_state.pos}, #prev={len(new_state.prev_positions)}')

      if state and state == new_state:
        state.prev_positions.append(prev_state.pos)

    def wait_then_move(cur_state, extra_wait_days=0):
      # Waiting with cooldown.
      if cur_state.cooldown >= 1:
        cur_state = self.wait_for_cooldown(cur_state,
                                           extra_wait_days=extra_wait_days)
        if cur_state is None:
          # if self.debug:
            # print(f' skip from cooldown')
          return

      assert cur_state.cooldown < 1
      for nb_cell in get_neighbour_positions(cur_state.pos, self.game_map,
                                             return_cell=True):
        # Can not go pass through enemy citytile.
        if cell_has_target_player_citytile(nb_cell, self.opponent):
          continue

        # TODO: maybe not in cooldown?
        # Skip enemy unit in cooldown.
        if (nb_cell.unit and nb_cell.unit.team == self.opponent.team):
            # nb_cell.unit.team == self.game.opponent_id and not nb_cell.unit.can_act()):
          continue

        next_state = self.move(cur_state, nb_cell)
        if not next_state:
          # if self.debug:
            # print(f' skip from move: cur_pos={cur_state.pos} to {nb_cell.pos}')
          continue
        heap_push(cur_state, next_state)

    upkeep = get_unit_upkeep(self.worker)
    start_state = SearchState(self.turn, self.worker.cell, self.worker.cargo,
                              int(self.worker.cooldown))
    start_state.arrival_fuel = cargo_total_fuel(self.worker.cargo)
    heap_push(None, start_state)
    while q:
      cur_state = heapq.heappop(q)
      # if self.debug:
        # print(f' visting: {cur_state.pos} turn={cur_state.turn}')
      if cur_state.deleted:
        continue

      # Do not leave citytile.
      is_player_citytile = cell_has_target_player_citytile(cur_state.cell,
                                                           self.player)
      if (self.not_leaving_citytile and cur_state != start_state
          and is_player_citytile):
        continue

      wait_then_move(cur_state)

      # TODO: test wait in the days
      # Wait during the night.
      if is_player_citytile and is_night(cur_state.turn):
        cooldown_wait_days = int(cur_state.cooldown)
        days_till_next_day = CIRCLE_LENGH - (cur_state.turn % CIRCLE_LENGH)
        extra_wait_days = max(cooldown_wait_days, days_till_next_day) - cooldown_wait_days
        wait_then_move(cur_state, extra_wait_days)

    if self.debug:
      for y in range(self.game_map.height):
        for x in range(self.game_map.width):
          st = self.state_map[x][y]
          if st:
            a = annotate.text(x, y, f'{st.turn-self.turn}', fontsize=32)
            self.actions.append(a)

            # a = annotate.text(st.pos.x, st.pos.y, f'{len(st.prev_positions)}', fontsize=32)
            # self.actions.extend([a])

            # prev = st.prev
            # if prev:
              # line = annotate.line(prev.pos.x, prev.pos.y,
                                  # st.pos.x, st.pos.y)
              # self.actions.append(line)
            # for prev in st.prev_positions:
              # if prev:
                # line = annotate.line(prev.pos.x, prev.pos.y,
                                    # st.pos.x, st.pos.y)
                # self.actions.append(line)


  def query_dest_turns(self, pos):
    state = self.state_map[pos.x][pos.y]
    if state is None:
      return MAX_PATH_WEIGHT
    return state.turn - self.turn

  # TODO: debug the new search algorithm (with path points)
  def get_next_step_path_points(self, target_pos, worker_pos):
    self.actions.clear()

    st = self.state_map[target_pos.x][target_pos.y]
    next_step_path_points = set()

    # visited_states = {st.key}

    path_positions = {target_pos}
    q = deque([st])
    while q:
      st = q.popleft()
      # if self.debug:
        # print(f'search from {st.pos} t={st.turn}, #prev={len(st.prev_positions)}')
      if st is None:
        continue
      for prev_pos in st.prev_positions:
        prev = self.state_map[prev_pos.x][prev_pos.y]
        # Root
        if prev is None:
          continue

        # if self.debug:
          # tmp = self.state_map[prev.pos.x][prev.pos.y]
          # print(f' >> prev {prev.pos} t={prev.turn}, #prev={len(prev.prev_positions)} #prev={len(tmp.prev_positions)}')

        # Collect points of next step from worker position
        if prev.pos == worker_pos:
          next_step_path_points.add(st.pos)

        # if self.debug:
          # line = annotate.line(prev.pos.x, prev.pos.y,
                               # st.pos.x, st.pos.y)
          # self.actions.extend([line])

        # Added by other node.
        if prev.pos in path_positions:
          # if self.debug:
            # print(f' >> prev {prev.pos} t={prev.turn} in path_positions, skip')
          continue

        # if self.debug:
          # print(f' >> adding {prev.pos} t={prev.turn} to queue and path_positions')
        q.append(prev)
        path_positions.add(prev.pos)


        # line = annotate.line(prev.pos.x, prev.pos.y, st.pos.x, st.pos.y)
        # self.actions.append(line)
    return next_step_path_points
    # return path_positions


    path_positions = {target_pos}
    st = self.state_map[target_pos.x][target_pos.y]
    while st:
      path_positions.add(st.pos)
      st = st.prev
    return path_positions



# TODO: add near resource tile to cluster
class ClusterInfo:

  def __init__(self, game):
    self.game = game
    self.game_map = game.game_map
    self.position_to_cid = np.ones((self.game_map.width, self.game_map.height)) * -1
    self.cluster_fuel = {}  # (cluster_id, cluster fuel)
    self.oppopent_citytile_count = defaultdict(int)
    self.player_citytile_count = defaultdict(int)
    self.max_cluster_fuel = 0
    self.max_cluster_id = 0

  def set_cid(self, pos, cid):
    self.position_to_cid[pos.x][pos.y] = cid

  def get_cid(self, pos):
    return self.position_to_cid[pos.x][pos.y]

  def compute(self):
    def is_valid_resource_cell(cell):
      return cell.has_resource()

    def search_cluster(start_cell):
      opponent_citytile_positions = set()
      player_citytile_positions = set()
      total_fuel = resource_fuel(start_cell.resource)
      q = deque([start_cell.pos])
      self.set_cid(start_cell.pos, max_cluster_id)

      while q:
        cur = q.popleft()
        cur_cell = self.game_map.get_cell_by_pos(cur)
        for nb_cell in get_neighbour_positions(cur, self.game_map, return_cell=True):
          newpos = nb_cell.pos

          # Count citytile.
          if nb_cell.citytile != None:
            if cell_has_opponent_citytile(nb_cell, self.game):
              opponent_citytile_positions.add(nb_cell.pos)
            else:
              assert cell_has_player_citytile(nb_cell, self.game)
              player_citytile_positions.add(nb_cell.pos)

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

      return (total_fuel,
              len(player_citytile_positions),
              len(opponent_citytile_positions))

    max_cluster_id = 0

    for x in range(self.game_map.width):
      for y in range(self.game_map.height):
        pos = Position(x, y)
        if self.get_cid(pos) >= 0:
          continue

        cell = self.game_map.get_cell_by_pos(pos)
        if not is_valid_resource_cell(cell):
          continue

        fuel, n_player_citytile, n_opponent_citytile = search_cluster(cell)
        self.cluster_fuel[max_cluster_id] = fuel
        self.player_citytile_count[max_cluster_id] = n_player_citytile
        self.oppopent_citytile_count[max_cluster_id] = n_opponent_citytile
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
    return self.game.circle_turn

  @property
  def round_factor(self):
    return 1 - self.game.days_this_round / DAY_LENGTH

  @property
  def game_map(self):
    return self.game.game_map

  @property
  def player(self):
    return self.game.player

  def player_available_workers(self):
    workers = [unit for unit in self.game.player.units
               if (unit.is_worker() and not unit.has_planned_action)]
    return workers

  def update(self, observation, configuration):
    self.game.update(observation, configuration)

    # Clear up actions for current step.
    self.actions = []

  def add_unit_action(self, unit, action):
    assert unit.has_planned_action == False

    unit.has_planned_action = True
    self.actions.append(action)


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

    # def nearest_opponent_worker_dist(cell):



    # TODO: ref citytile to city
    for y in range(self.game.map_height):
      for x in range(self.game.map_width):
        cell = self.game_map.get_cell(x, y)
        cell.unit = None
        cell.has_buildable_neighbour = False

        cell.is_coal_target = False
        cell.is_uranium_target = False
        if cell.has_resource():
          cell.is_coal_target = is_resource_coal(cell.resource)
          cell.is_uranium_target = is_resource_uranium(cell.resource)

        cell.is_near_resource = self.is_near_resource_cell(cell)
        cell.n_citytile_neighbour = self.count_citytile_neighbours(cell)


    for unit in self.game.player.units:
      cell = self.game_map.get_cell_by_pos(unit.pos)
      cell.unit = unit

    for unit in self.game.opponent.units:
      cell = self.game_map.get_cell_by_pos(unit.pos)
      cell.unit = unit

  def update_unit_info(self):
    self.quickest_path_pairs = {}
    empty_set = set()
    for unit in self.game.player.units:
      unit.cell = self.game_map.get_cell_by_pos(unit.pos)
      unit.has_planned_action = False
      unit.target_pos = unit.pos
      unit.target = self.game_map.get_cell_by_pos(unit.pos)
      unit.target_score = 0
      # unit.target_pos = None
      # unit.target = None

      unit.transfer_build_locations = set()
      unit.target_cluster_id = -1
      unit.is_cluster_owner = False
      unit.is_transfer_worker = False
      unit.target_city_id = None

      # TODO: it should not be used?
      unit.surviving_turns = unit_surviving_turns(self.game.turn, unit)

      unit.unit_night_count = cargo_night_endurance(unit.cargo, get_unit_upkeep(unit))

      left_turns_this_round = get_left_turns_this_round(self.game.turn)
      unit.is_cargo_not_enough_for_nights = unit.surviving_turns < left_turns_this_round

      # round_night_count = get_night_count_this_round(self.game.turn)
      # dying = unit.unit_night_count < round_night_count
      # assert dying == unit.is_cargo_not_enough_for_nights

      debug = False

      debug = (unit.id in DRAW_UNIT_LIST and DRAW_QUICK_PATH_VALUE)
      quickest_path = QuickestPath(self.game, unit, debug=debug)
      quickest_path.compute()
      self.actions.extend(quickest_path.actions)


      # debug = (unit.id in DRAW_UNIT_LIST and DRAW_QUICK_PATH_VALUE)
      quickest_path_wo_citytile = QuickestPath(self.game, unit,
                                               not_leaving_citytile=True, debug=debug)
      quickest_path_wo_citytile.compute()
      # self.actions.extend(quickest_path_wo_citytile.actions)

      self.quickest_path_pairs[unit.id] = (quickest_path, quickest_path_wo_citytile)

      unit.cid_to_tile_pos = {}
      unit.cid_to_cluster_turns = {}

    for unit in self.game.opponent.units:
      unit.cell = self.game_map.get_cell_by_pos(unit.pos)
      quickest_path = QuickestPath(self.game, unit, debug=debug)
      quickest_path.compute()
      self.quickest_path_pairs[unit.id] = (quickest_path, None)

  @functools.lru_cache(maxsize=1024, typed=False)
  def get_cell_opponent_unit_min_arrival_turns(self, cell):
    min_turns = MAX_PATH_WEIGHT
    min_turn_cnt = 0
    for unit in self.game.opponent.units:
      path, _ = self.quickest_path_pairs[unit.id]
      arrival_turns = path.query_dest_turns(cell.pos)
      if arrival_turns < min_turns:
        min_turn_cnt = 1
        min_turns = arrival_turns
      elif arrival_turns == min_turns:
        min_turn_cnt += 1
    return min_turns, min_turn_cnt

  def count_citytile_neighbours(self, cell, min_citytile=1):
    cnt = 0
    for nb_cell in get_neighbour_positions(cell.pos, self.game_map, return_cell=True):
      if cell_has_player_citytile(nb_cell, self.game):
        cnt += 1
    return cnt

  def is_near_resource_cell(self, cell):
    def has_resource_tile_neighbour(cell):
      for nb_cell in get_neighbour_positions(cell.pos, self.game_map, return_cell=True):
        if (nb_cell.has_resource()):
          # Set resource type for cell based on neighbour cell
          if is_resource_coal(nb_cell.resource):
            cell.is_coal_target = True
          elif is_resource_uranium(nb_cell.resource):
            cell.is_uranium_target = True
          return True
      return False

    return (not cell.has_resource()
            and cell.citytile is None
            and has_resource_tile_neighbour(cell))

  def assign_worker_target(self):
    MAX_UNIT_PER_CITY = 8
    g = self.game
    player = g.player

    workers = self.player_available_workers()

    def cell_has_buildable_neighbour(cell):
      for nb_cell in get_neighbour_positions(cell.pos, self.game_map, return_cell=True):
        if (not nb_cell.has_resource()
            and nb_cell.citytile is None):
          return True
      return False


    def collect_target_cells():
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
          elif cell.n_citytile_neighbour > 0:
            is_target_cell = True
          elif (cell.unit and cell.unit.team == player.team):
            is_target_cell = True

          if is_target_cell:
            cell.has_buildable_neighbour = cell_has_buildable_neighbour(cell)

            targets = [cell]
            # if cell.is_coal_target or cell.is_uranium_target:
              # targets = [cell] * 2
            target_cells.extend(targets)
      return target_cells


    def is_deficient_resource_tile(resource_tile):
      if not is_night(self.game.turn):
        return False

      if not resource_tile.has_resource():
        return True

      if (is_resource_wood(resource_tile.resource)
          and resource_tile.resource.amount < 60):
        return True
      return False


    def is_resource_tile_can_save_dying_worker(resource_tile, worker, arrival_turns):
      if (is_worker_on_last_city_tiles(worker)
          and is_deficient_resource_tile(resource_tile)):
        return False

      # After arrival, resource is enough to reach the next circle.
      def estimate_resource_night_count(cell, upkeep):
        if not cell.has_resource():
          return 0
        wait_turns = resource_researched_wait_turns(cell.resource, player, arrival_turns,
                                            surviving_turns=worker.surviving_turns)
        if wait_turns < 0:
          return 0
        cargo = resource_to_cargo(cell.resource)
        return cargo_night_endurance(cargo, upkeep)

      def estimate_cell_night_count(cell, upkeep, game_map):
        nights = estimate_resource_night_count(cell, upkeep)
        for nb_cell in get_neighbour_positions(cell.pos, game_map,
                                               return_cell=True):
          nights += estimate_resource_night_count(nb_cell, upkeep)
        return nights


      if worker.is_cargo_not_enough_for_nights:
        cell_nights = estimate_cell_night_count(resource_tile,
                                                get_unit_upkeep(worker),
                                                g.game_map)
        round_nights = get_night_count_this_round(g.turn)
        if worker.unit_night_count + cell_nights >= round_nights:
          # if worker.id in DRAW_UNIT_LIST and resource_tile.pos in [Position(1, 10)]:
            # cell_nights = estimate_cell_night_count(resource_tile,
                                                    # get_unit_upkeep(worker),
                                                    # g.game_map)
            # round_nights = get_night_count_this_round(g.turn)
            # print(f' dying={worker.is_cargo_not_enough_for_nights}, unit_last={worker.unit_night_count}, cell_nights={cell_nights}, round_nights={round_nights}')
            # print(f' not_has_resource={not resource_tile.has_resource()}')
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

    DEFAULT_RESOURCE_WT = 1
    def get_resource_weight(worker, resource_tile, arrival_turns):
      # Give a small weight for any resource 0.1 TODO: any other option?
      wt = 0

      if is_deficient_resource_tile(resource_tile):
        # do not goto resource tile at night if there is not much.
        fuel_wt = 0
      elif (is_resource_wood(resource_tile.resource)
          and not resource_tile.has_buildable_neighbour):
        # For wood cell with no buildable neightbor, demote its weight
        # TODO: why it's cell resource value here?
        _, fuel_wt = get_cell_resource_values(resource_tile, player,
                                           unit=worker,
                                           move_days=arrival_turns,
                                           surviving_turns=worker.surviving_turns)
        fuel_wt /= 2
        # if resource_tile.pos == Position(5, 28):
          # print(f'triggered demote for no neighbour c[{resource_tile.pos}]')
      else:
        _, fuel_wt = get_one_step_collection_values(resource_tile, player, self.game,
                                                 unit=worker,
                                                 move_days=arrival_turns,
                                                 surviving_turns=worker.surviving_turns)
      if worker.id in DRAW_UNIT_LIST and resource_tile.pos in MAP_POS_LIST:
        print(f"{resource_tile.pos} 1. fuel_wt={fuel_wt}")
      if fuel_wt:
        wt += DEFAULT_RESOURCE_WT

      #TODO: encourage worker into near/resource tile, not necessary dying
      # Try to hide next to resource grid in the night.
      if is_resource_tile_can_save_dying_worker(resource_tile, worker, arrival_turns):
        wt += UNIT_SAVED_BY_RES_WEIGHT

      if worker.id in DRAW_UNIT_LIST and resource_tile.pos in MAP_POS_LIST:
        print(f"{resource_tile.pos} 2. wt={wt}")

        # if worker.is_cargo_not_enough_for_nights and g.turn > 65:
          # wt += 1000

      # TODO: Consider drop cluster boosting when dist <= 1
      cid = self.cluster_info.get_cid(resource_tile.pos)
      boost_cluster = 0
      if (worker.target_cluster_id >= 0 and worker.target_cluster_id == cid):
          # worker.cid_to_cluster_turns[cid] > 1):
        cluster_fuel_factor = self.cluster_info.query_cluster_fuel_factor(resource_tile.pos)
        boost_cluster += CLUSTER_BOOST_WEIGHT * cluster_fuel_factor

      # if worker.id == 'u_8' and resource_tile.pos in [Position(5, 24), Position(6, 23)]:
        # print(f"t={g.turn},cell={resource_tile.pos}, wt={wt}, wt_c={boost_cluster}")

      # Do not boost cluster at night
      # if self.game.is_night:
        # boost_cluster = 0

      opponent_weight = 0
      if fuel_wt > 0:
        min_turns, min_turn_cnt = self.get_cell_opponent_unit_min_arrival_turns(resource_tile)
        opponent_weight = min_turn_cnt / (min_turns or 1)
      return (wt / (arrival_turns + 1)
              + boost_cluster
              + fuel_wt
              + opponent_weight)
      # return wt / dist_decay(dist, g.game_map) + boost_cluster
      # return wt / (dist + 0.1) + boost_cluste1
      # return wt / dist_decay(dist, g.game_map)

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

      wt = 0.01

      # [1] Stay at city will gain this amout of fuel
      # amount, fuel = get_one_step_collection_values(citytile.cell, player,
                                                    # self.game)
      # if self.game.is_night:
        # wt += max(fuel - LIGHT_UPKEEP['CITY'], 0)
        # wt += city_last_nights(city)

      # if worker.id in DRAW_UNIT_LIST and city_cell.pos in MAP_POS_LIST:
        # print(f"city_crash_boost 0:wt={wt}")

      #TODO: add property to city
      # [2] Try to hide in the city if worker will run out of fuel at night
      city_wont_last = city_wont_last_at_nights(g.turn, city)
      city_last = not city_wont_last
      if worker.is_cargo_not_enough_for_nights:
        # Hide in the city
        if city_last:
          wt += max(min(city_last_nights(city) / 10, 2), 0) * self.round_factor
        # elif cargo_total_amount(worker.cargo) == 0:
          # Escape from dying city.
          # wt = -99999
          # return wt

      # if worker.id in DRAW_UNIT_LIST and city_cell.pos in MAP_POS_LIST:
        # print(f"city_crash_boost 1:wt={wt}, [{city_cell.pos}] city_last={city_last_nights(city)}")


      # TODO(wangfei): estimate worker will full
      n_citytile = len(city.citytiles)

      # [3] Try move worker onto city to save it
      # Problem: 1) what if fuel < min_fuel?
      #          2) what if we should move to citytile earlier?
      # min_fuel = 60#(self.game.is_day and 40 or 40)
      # min_city_last = 20 if g.turn < 320 else 10
      # TODO: is this round needed? or could be fine turned
      # if (self.circle_turn >= 20
          # and n_citytile >= 2
          # and city_last_nights(city) < min_city_last
          # and worker_total_fuel(worker) >= min_fuel
          # or (is_worker_cargo_full(worker))):
        # wt += worker_total_fuel(worker)
        # if not city_wont_last:
          # wt += worker_total_fuel(worker)

        # # Fuel city task.
        # self.worker_fuel_city_tasks.add((worker.id, city_cell.pos))

      _, quick_path = self.quickest_path_pairs[worker.id]
      arrival_turns_wo_city = quick_path.query_dest_turns(city_cell.pos)

      # city_last_turns = city_last_days(self.game.turn, city)
      # dest_state = quick_path.state_map[city_cell.pos.x][city_cell.pos.y]
      min_city_arrival_turns = worker_city_min_arrival_turns(worker, city)
      # if dest_state and worker.id in DRAW_UNIT_LIST and city_cell.pos in [Position(12, 14)]:
        # print(f' c{city_cell.pos}, w_fuel={worker_total_fuel(worker)} wont_last={city_wont_last}, arrival_turns={arrival_turns_wo_city}, city_last={city_last_turns}, dest_fuel={dest_state.arrival_fuel}, min_arrival_turns={min_city_arrival_turns}')
        # print(f' c{city_cell.pos}, w_fuel={worker_total_fuel(worker)} wont_last={city_wont_last}, arrival_turns={arrival_turns_wo_city}, city_last_nights={city_last_nights(city)}, dest_fuel={dest_state.arrival_fuel}, min_arrival_turns={min_city_arrival_turns}')


      city_crash_boost = 0
      city_last_turns = city_last_days(self.game.turn, city)

      # If the worker is a wood full worker, goto nearest city tiles when possible.
      if (city_wont_last
          and arrival_turns <= city_last_turns
          and worker.cargo.wood == WORKER_RESOURCE_CAPACITY
          and not worker.is_cluster_owner):
        wt += 200

      # If a worker can arrive at this city with some min fuel (or full cargo)
      if city_wont_last and n_citytile >= 2:
        _, quick_path = self.quickest_path_pairs[worker.id]
        arrival_turns_wo_city = quick_path.query_dest_turns(city_cell.pos)
        dest_state = quick_path.state_map[city_cell.pos.x][city_cell.pos.y]
        min_city_arrival_turns = worker_city_min_arrival_turns(worker, city)
        if (arrival_turns_wo_city == min_city_arrival_turns    # only goto nearest city tiles.
            and arrival_turns_wo_city <= city_last_turns):     # city should last when arrived.
          not_full_woker_goto_city = (
              city_last_turns - arrival_turns_wo_city <= 6   # do not goto city too earlier.
              and dest_state.arrival_fuel >= 60
          )
          full_worker_goto_city = (
            worker.get_cargo_space_left() == 0
            and (worker.cargo.coal > 0 or worker.cargo.uranium > 0)
          )
          if (not_full_woker_goto_city or full_worker_goto_city):
            city_crash_boost += n_citytile * max(CITYTILE_LOST_WEIGHT, worker_total_fuel(worker))
            # city_crash_boost = max(CITYTILE_LOST_WEIGHT, worker_total_cargo(worker))
            # Fuel city task.
            self.worker_fuel_city_tasks.add((worker.id, city_cell.pos))

            # if worker.id in DRAW_UNIT_LIST and city_cell.pos in MAP_POS_LIST:
              # print(f"city_crash_boost 2: {city_crash_boost}, wt={wt}")

      # TODO-IMPORTANT: maybe go back to cargo, to make it transfer to other worker.
      # THIS IS: we should distribute resource evenly for all city tiles.
      # If worker if full, go back to city.
      # if worker.id in DRAW_UNIT_LIST and city_cell.pos in [Position(9, 8), Position(9, 9)]:
        # print(f' c{city_cell.pos}, w_fuel={worker_total_fuel(worker)} wont_last={city_wont_last}, arrival_turns={arrival_turns_wo_city}, city_last={city_last_turns},  min_arrival_turns={min_city_arrival_turns}')
        # min_city_last = 20 if g.turn < 320 else 10
        # D = int((self.game.turn + city_last_turns) < MAX_DAYS)
        # print(f' c{city_cell.pos}, A={not city_wont_last}, B={arrival_turns_wo_city <= min_city_arrival_turns + 2}, C={worker_total_cargo(worker) > 85}, D={D}')
      is_worker_full = worker_total_cargo(worker) >= 85
      no_resoruce_on_map = (self.cluster_info.max_cluster_fuel == 0)
      if (arrival_turns_wo_city == min_city_arrival_turns
          and (is_worker_full or no_resoruce_on_map)
          and (self.game.turn + city_last_turns) < MAX_DAYS):
          # and self.game.turn + city_last_turns < MAX_DAYS):
        # wt += worker_total_fuel(worker) * n_citytile
        if n_citytile >= 2:
          city_crash_boost += worker_total_fuel(worker) * n_citytile
          self.worker_fuel_city_tasks.add((worker.id, city_cell.pos))

        if n_citytile == 1 and no_resoruce_on_map:
          city_crash_boost += 0.2
          self.worker_fuel_city_tasks.add((worker.id, city_cell.pos))


      # if worker.id in DRAW_UNIT_LIST and city_cell.pos in MAP_POS_LIST:
        # print(f"city_crash_boost 3: {city_crash_boost}, wt={wt}")

      # Boost when worker has resource and city tile won't last.
      days_left = DAY_LENGTH - self.circle_turn
      round_nights = get_night_count_this_round(g.turn)

      # [not used] Boost based on woker, city assignment.
      if (worker.target_city_id == citytile.cityid
          and worker.pos.distance_to(city_cell.pos) == 1):
        wt += 1000 * n_citytile

        # More weights on dying city
        # if city_left_days < round_nights:
          # wt += n_citytile * CITYTILE_LOST_WEIGHT

      # When there is no fuel on the map, go back to city
      # if self.cluster_info.max_cluster_fuel == 0:
        # wt += 100

      # TODO(wangfei): try delete this rule
      # TODO: support night
      # Save more citytile if worker has enough resource to save it
      # - enough time to arrive
      # - with substential improvement of its living
      # unit_time_cost = (dist - 1) * get_unit_action_cooldown(worker) + 1
      # city_left_days = math.floor(city.fuel / city.light_upkeep)
      # woker_fuel = worker_total_fuel(worker)
      # city_left_days_deposited = math.floor((city.fuel + woker_fuel) / city.light_upkeep)
      # if (days_left >= arrival_turns
          # and city_left_days < NIGHT_LENGTH
          # and city_left_days_deposited > NIGHT_LENGTH
          # and city_left_days_deposited - city_left_days >= (18 // n_citytile)):
        # wt += CITYTILE_LOST_WEIGHT * len(city.citytiles)

      # if worker.id == 'u_7' and city.id in ['c_12']:
        # print(f"t={g.turn}, {worker.id}, nc={worker.surviving_turns}, tnc={arrival_turns} tile={citytile.pos}, wt={wt}, dying={worker.is_cargo_not_enough_for_nights}")


      # Overwrite arrival_turns
      if self.is_worker_fuel_city_task(worker, city_cell.pos):
        _, quick_path = self.quickest_path_pairs[worker.id]
        arrival_turns = quick_path.query_dest_turns(city_cell.pos)
        if arrival_turns >= MAX_PATH_WEIGHT:
          return -9999

      # if worker.id in DRAW_UNIT_LIST and city_cell.pos in MAP_POS_LIST:
        # print(f" wt={wt}, city_crash_boost={city_crash_boost}")
      # return (wt + city_crash_boost) / (arrival_turns + 1)
      return (wt) / (arrival_turns + 1) + (city_crash_boost / (arrival_turns / 5 + 1))
      # return (wt + ) / (arrival_turns + 1) + city_crash_boost
      # return wt / dist_decay(dist, g.game_map)
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

    self.worker_build_city_tasks = set()
    self.worker_fuel_city_tasks = set()

      # TODO(wangfei): merge near resource tile and resource tile weight functon
    def get_near_resource_tile_weight(worker, near_resource_tile, arrival_turns):
      wt = 0
      # Not using unit so near resource tile will have more weight over resource tile.
      amount, fuel_wt = get_one_step_collection_values(near_resource_tile, player,
                                                    self.game,
                                                    move_days=arrival_turns,
                                                    surviving_turns=worker.surviving_turns)
      fuel_wt /= 3  # Use smaller weight for near resource tile
      if fuel_wt > 0:
        wt += DEFAULT_RESOURCE_WT / 2

      if near_resource_tile.pos in MAP_POS_LIST:
        print(f'c[{near_resource_tile.pos}] @1, wt={wt}')

      # Boost the target collect amount by 2 (for cooldown) to test for citytile building.
      # it's 2, because one step move and one step for cooldown
      build_city_bonus = False
      days_left = BUILD_CITYTILE_ROUND - self.circle_turn
      if (self.circle_turn < BUILD_CITYTILE_ROUND
          # and worker_enough_cargo_to_build(worker, amount*2)
          and worker_enough_cargo_to_build(worker, amount*2)
          and days_left >= arrival_turns):
        build_city_bonus = True

      # To build on transfer build location.
      # print(f'w[{worker.id}], transfer_build_locations={worker.transfer_build_locations}')
      if (near_resource_tile.pos in worker.transfer_build_locations):
        build_city_bonus = True

      boost_cluster = 0
      is_next_to_target_cluster = cell_next_to_target_cluster(worker, near_resource_tile)
      if is_next_to_target_cluster:
        cid = worker.target_cluster_id
        cluster_fuel_factor = self.cluster_info.cluster_fuel[cid] / self.cluster_info.max_cluster_fuel
        boost_cluster += CLUSTER_BOOST_WEIGHT * cluster_fuel_factor

      # Goto cluster has higher priority than city tile building.
      # if not is_next_to_target_cluster and worker.target_cluster_id >= 0:
        # build_city_bonus = False

      # Boost worker into city build position when night left = 4 for first city building
      # 1) circle_round == 36 (DAY_LENGTH - cooldown * 2)
      # 2) dist == 1
      # 3) worker.surviving_turns >= 4
      # if (self.circle_turn == DAY_LENGTH - get_unit_action_cooldown(worker) * 2
          # and arrival_turns == 1
          # and worker.surviving_turns >= self.game.night_in_round):
        # build_city_bonus = True

      # Overwrite arrival_turns or not boost city building.
      if self.is_worker_building_citytile(worker, near_resource_tile.pos):
        _, quick_path = self.quickest_path_pairs[worker.id]
        tmp_arrival_turns = quick_path.query_dest_turns(near_resource_tile.pos)
        if tmp_arrival_turns >= MAX_PATH_WEIGHT:
          build_city_bonus = False
        else:
          arrival_turns = tmp_arrival_turns

      # Do not boost cluster owner to build city at night
      # TODO: replace with focus cluster owner
      if worker.target_cluster_id >= 0 and self.circle_turn >= 28:
        build_city_bonus = False

      # TODO: test this threshold.
      # Too large the build city bonus will cause worker divergence from its coal mining position
      if (build_city_bonus and arrival_turns <= 3
          and not is_first_night(self.game.turn)):
        wt += 1000

        # Encourage worker to build connected city tiles.
        if near_resource_tile.n_citytile_neighbour > 0:
          wt += 100 * near_resource_tile.n_citytile_neighbour

        # mark build city cell
        self.worker_build_city_tasks.add((worker.id, near_resource_tile.pos))

      if near_resource_tile.pos in MAP_POS_LIST:
        print(f'c[{near_resource_tile.pos}] @1, wt={wt}, arrival_turns={arrival_turns}')

      # Try to hide next to resource grid.
      if is_resource_tile_can_save_dying_worker(near_resource_tile, worker,
                                                arrival_turns):
        wt += UNIT_SAVED_BY_RES_WEIGHT
          # wt += 1000 / 2

      # if worker.id == 'u_3' and near_resource_tile.pos in [Position(5, 28), Position(8, 24),
                                                           # Position(4, 27)]:
        # print(f"t={g.turn}, near_resource_cell={near_resource_tile.pos}, dist={dist}, wt={wt}, wt_c={boost_cluster}, amt={amount}, fuel_wt=fuel_wt}")
      # if self.game.is_night:
        # boost_cluster = 0

      # TODO: this can be used to defend my tile.
      opponent_weight = 0
      if fuel_wt > 0:
        min_turns, min_turn_cnt = self.get_cell_opponent_unit_min_arrival_turns(near_resource_tile)
        opponent_weight = min_turn_cnt / (min_turns or 1)

      # return wt / (arrival_turns + 1) + boost_cluster / (arrival_turns // 2 + 1)
      return (wt / (arrival_turns + 1)
              + boost_cluster
              + fuel_wt
              + opponent_weight)
      # return wt / dist_decay(dist, g.game_map) + boost_cluster
      # return wt / (dist + 0.1) + boost_cluster

    def get_city_tile_neighbour_weight(worker, cell):
      wt = -999
      # If only do one transfer, then need to sort the positions
      if cell.pos in worker.transfer_build_locations:
        nb_citytiles = [nb_cell.citytile
                        for nb_cell in get_neighbour_positions(cell.pos, self.game_map, return_cell=True)
                        if cell_has_player_citytile(nb_cell, self.game)]
        assert len(nb_citytiles) > 0
        wt = CLUSTER_BOOST_WEIGHT * 2 * len(nb_citytiles)

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
        wt = 0.001
      return wt



    # Value matrix for ship target assginment
    # * row: workers
    # * column: point of interests
    target_cells = collect_target_cells()
    weights = np.ones((len(workers), len(target_cells))) * -9999
    for i, worker in enumerate(workers):
      for j, target in enumerate(target_cells):
        # Can't arrive at target cell.
        quicker_path, quicker_dest_turns = self.select_quicker_path(worker, target.pos)
        if quicker_dest_turns >= MAX_PATH_WEIGHT:
          # if worker.id in DRAW_UNIT_LIST and target.pos in MAP_POS_LIST:
            # v = -9999
            # print(f"w[{worker.id}], cd={worker.cooldown}, t[{target.pos}], wt={v:.1f}", file=sys.stderr)
          continue

        # TODO: drop
        # Other worker can't move onto worker cell with cd > 1
        if target.unit and target.unit.cooldown >= 1 and worker.id != target.unit.id:
          continue

        if cell_has_player_citytile(target, self.game):
          # TODO: when should we use path without citytile?
          # _, path_wo_cc = self.quickest_path_pairs[worker.id]
          # dest_turns = path_wo_cc.query_dest_turns(target.pos)
          # if dest_turns >= MAX_PATH_WEIGHT:
            # continue
          # v = get_city_tile_weight(worker, target, dest_turns)
          v = get_city_tile_weight(worker, target, quicker_dest_turns)
          # if worker.id in DRAW_UNIT_LIST and target.pos in MAP_POS_LIST:
            # print(f"to t[{target.pos}], v={v}, arr={quicker_dest_turns}", file=sys.stderr)
        elif target.has_resource():
          v = get_resource_weight(worker, target, quicker_dest_turns)
        elif target.is_near_resource:
          # _, path_wo_cc = self.quickest_path_pairs[worker.id]
          # dest_turns = path_wo_cc.query_dest_turns(target.pos)
          # if dest_turns >= MAX_PATH_WEIGHT:
            # continue
          # v = get_near_resource_tile_weight(worker, target, dest_turns)
          #TODO: use path_wo_cc when build tiles
          v = get_near_resource_tile_weight(worker, target, quicker_dest_turns)
        elif target.n_citytile_neighbour > 0:
          v = get_city_tile_neighbour_weight(worker, target)
        else:
          v = get_worker_tile_weight(worker, target)

        weights[i, j] = v


        if worker.id in DRAW_UNIT_LIST and DRAW_UNIT_TARGET_VALUE:
          pos = target.pos
          a = annotate.text(pos.x, pos.y, f'{v:.2f}', fontsize=32)
          self.actions.append(a)

        # if worker.id == 'u_48' and target.pos in [Position(18, 9), Position(18, 11),]:
          # print(f"w[{worker.id}], cd={worker.cooldown}, t[{target.pos}], wt={v:.1f}", file=sys.stderr)
        # if worker.id == 'u_26' and target.pos in [Position(8, 9), Position(8, 8),]:
          # print(f"w[{worker.id}], cd={worker.cooldown}, t[{target.pos}], wt={v:.1f}", file=sys.stderr)

    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    for worker_idx, target_idx in zip(rows, cols):
      if weights[worker_idx, target_idx] < 0:
        continue

      worker = workers[worker_idx]
      worker.target_score = weights[worker_idx, target_idx]
      target = target_cells[target_idx]
      worker.target = target
      worker.target_pos = target.pos

      if DRAW_UNIT_ACTION:
        # if worker.id in ['u_48', 'u_26']:
          # print(f'{worker.id}, v={weights[worker_idx, target_idx]}, target={worker.target.pos}',
                # file=sys.stderr)
        x = annotate.x(worker.pos.x, worker.pos.y)
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

    MAX_MOVE_WEIGHT = 99999
    def get_step_weight(worker, next_position, quick_path, path_positions):
      """Only workers next 5 positions will be computed here."""
      # if worker.pos == next_position:
        # If a worker can't move, it's a stone.
        # if not worker.can_act():
          # return 1000

        # If woker is building city.
        # if worker.is_building_city:
          # return 1000

      assert worker.target_pos is not None
      assert path_positions is not None

      next_cell = g.game_map.get_cell_by_pos(next_position)
      # Can't move on opponent citytile.
      if (next_cell.citytile is not None
          and next_cell.citytile.team == self.game.opponent.team):
        return -MAX_MOVE_WEIGHT

      # Do not move onto a transfer worker
      if (next_cell.unit
          and next_cell.unit.team == player.team
          and (next_cell.unit.is_transfer_worker
               or not next_cell.unit.can_act())):
        return -MAX_MOVE_WEIGHT


      v = 0
      # Stay at current position.
      if next_position == worker.pos:
        v += 1

      if (next_position == worker.target_pos or
          (next_position != worker.pos and next_position in path_positions)):
        v += (worker.target_score + 10)
        # v += 50

        # # Priority all positions of th dying worker, let others make room for him.
        # if worker.is_cargo_not_enough_for_nights:
          # v += 100

        # target_cell = self.game_map.get_cell_by_pos(worker.target_pos)
        # if target_cell.has_resource():
          # v += 1

        # # TODO: this may not be needed
        # if (target_cell.is_near_resource
            # and cell_has_player_citytile(next_cell, self.game)):
          # v -= 5

        # # Try step on resource: the worker version is better, maybe because
        # # other worker can use that.
        # amount, fuel = get_cell_resource_values(next_cell, g.player)
        # if fuel > 0:
          # v += 1

        if next_position in worker.transfer_build_locations:
          v += 1000


      if worker.id in DRAW_UNIT_LIST and DRAW_UNIT_MOVE_VALUE:
        a = annotate.text(next_position.x, next_position.y, f'{int(v)}', fontsize=32)
        self.actions.append(a)

      # if worker.id in DRAW_UNIT_LIST:
        # print(f"w[{worker.id}]@{worker.pos}, next[{next_position}], v={v}, target={worker.target_pos}, next_points={path_positions}", file=sys.stderr)
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
      assert worker.target is not None

      quick_path, _ = self.select_quicker_path(worker, worker.target_pos)
      target_pos = worker.target.pos
      if (cell_has_player_citytile(worker.target, self.game)
          or self.is_worker_building_citytile(worker, target_pos)
          or self.is_worker_fuel_city_task(worker, target_pos)):
          # or worker.target.is_near_resource):
        # Use path no passing citytile
        _, quick_path = self.quickest_path_pairs[worker.id]
        # if worker.id in ['u_2']:
          # a, b = self.quickest_path_pairs[worker.id]
          # print(f' a={a.not_leaving_citytile}, b={b.not_leaving_citytile}')

      # if worker.id in ['u_2']:
        # print(f'-> {worker.id}, amt={worker.cargo}, building_tile: {is_worker_building_citytile(worker)}')
        # if is_worker_building_citytile(worker):
          # print(f'not leave city{quick_path.not_leaving_citytile}, path_positions={quick_path.get_next_step_path_points(worker.target_pos)}')

      # path staths from dest to start point
      # if worker.target_pos is None:
        # print(f'  w[{worker.id}]={worker.pos}, target={worker.target_pos}, S_path={path_positions}', file=sys.stderr)

      path_positions = quick_path.get_next_step_path_points(worker.target_pos, worker.pos)
      # if worker.id in DRAW_UNIT_LIST:
        # print(f'path_positions: {path_positions}')
      self.actions.extend(quick_path.actions)

      for next_position in gen_next_positions(worker):
        wt = get_step_weight(worker, next_position, quick_path, path_positions)
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

        # Do not build single city on first night:
        if (is_first_night(self.game.turn) and cell.n_citytile_neighbour == 0):
          continue

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
      if cur_research_points < URANIUM_RESEARCH_POINTS:
        cur_research_points += 1
        self.actions.append(citytile.research())



  def update_player_info(self):
    # Estimate number of research point in left day times.
    # n_city = len(self.player.cities)
    self.player.avg_research_point_growth = self.player.city_tile_count / CITY_ACTION_COOLDOWN

    opponent = self.game.opponent
    opponent.avg_research_point_growth = opponent.city_tile_count / CITY_ACTION_COOLDOWN

  def update_game_info(self):
    self.cluster_info = ClusterInfo(self.game)
    self.cluster_info.compute()

    self.update_player_info()
    self.update_game_map_info()

    # t1 = time.time()
    self.update_unit_info()
    # t2 = time.time()
    # print(f'....T_unit={t2-t1} seconds.')


  def assign_worker_to_resource_cluster(self, multi_worker=False):
    """For each no citytile cluster of size >= 2, find a worker with cargo space not 100.
      And set its target pos."""
    # Computes worker property.
    if self.game.player.city_tile_count < 2:
      return

    if self.cluster_info.max_cluster_id == 0:
      return

    def worker_cluster_turns_info(worker, cid):
      # TODO: also count as arrival for near_resource_tile
      x_pos, y_pos = np.where(self.cluster_info.position_to_cid == cid)
      n_resource_tile = len(x_pos)

      best_pos = None
      best_arrival_turns = MAX_PATH_WEIGHT
      for x, y in zip(x_pos, y_pos):
        pos = Position(x, y)
        quick_path, arrival_turns = self.select_quicker_path(worker, pos)
        if arrival_turns >= best_arrival_turns:
          continue

        best_pos = pos
        best_arrival_turns = arrival_turns

      # if self.game.turn == 15:
        # print(f"{worker.id}, target_tile[{best_pos}, dist={best_arrival_turns}, unit_nights={worker.surviving_turns}], arrival_turns={arrival_turns}",
            # file=sys.stderr)

      # Count near resource tile as arrival
      if best_arrival_turns <= 1:
        best_arrival_turns = 0
      return best_arrival_turns, best_pos, n_resource_tile

    def get_cluster_weight(worker, cid):
      # TODO: MAYBE keep it?
      if worker.target_city_id:
        return -9999

      fuel = self.cluster_info.cluster_fuel[cid]
      arrival_turns, tile_pos, n_tile = worker_cluster_turns_info(worker, cid)

      # Save tile position
      worker.cid_to_tile_pos[cid] = tile_pos
      worker.cid_to_cluster_turns[cid] = arrival_turns

      if arrival_turns >= MAX_PATH_WEIGHT:
        return -9999


      # resource cluster not researched.
      cell = self.game_map.get_cell_by_pos(tile_pos)
      debug=False
      if worker and worker.id in DRAW_UNIT_LIST:
        debug = True


      MAX_WAIT_ON_CLUSTER_TURNS = 6
      wait_turns = resource_researched_wait_turns(cell.resource, self.player,
                                                  move_days=arrival_turns,
                                                  surviving_turns=worker.surviving_turns, debug=debug)
      if wait_turns < 0 or wait_turns > MAX_WAIT_ON_CLUSTER_TURNS:
        return -9999

      # TODO: do not ignore enemy cluster, but could try use edge count info.
      # n_oppo_citytile = self.cluster_info.oppopent_citytile_count[cid]
      # n_player_citytile = self.cluster_info.player_citytile_count[cid]
      # if n_oppo_citytile > 0 and n_player_citytile == 0:
        # return -9999

      # if n_tile <= 1:
        # return 1
      # wt = fuel / (arrival_turns + wait_turns + 1)
      wt = fuel / (1.2  ** (arrival_turns + wait_turns) + 1)
      # if debug:
        # print(f" {worker.id}, c@{tile_pos} fuel={fuel}, wait={wait_turns}, arrival_turns={arrival_turns}, wt={wt}")
      return wt

    def gen_resource_clusters():
      for cid in range(self.cluster_info.max_cluster_id):
        x_pos, y_pos = np.where(self.cluster_info.position_to_cid == cid)
        n_resource_tile = len(x_pos)
        cell = self.game_map.get_cell(x_pos[0], y_pos[0])
        # TODO(wangfei): remove this
        if (n_resource_tile <= 1
            and cell.resource.type == Constants.RESOURCE_TYPES.WOOD):
          continue
        yield cid

    RESOURCE_WORKER_RATIO = 3
    def gen_multi_worker_resource_clusters():
      for cid in range(self.cluster_info.max_cluster_id):
        x_pos, y_pos = np.where(self.cluster_info.position_to_cid == cid)
        n_resource_tile = len(x_pos)

        n_workers = int(math.ceil(n_resource_tile / RESOURCE_WORKER_RATIO))
        if n_workers > 0:
          for _ in range(n_workers):
            yield cid

    workers = self.player_available_workers()
    if multi_worker:
      workers = [w for w in workers if w.target_cluster_id < 0]

    resource_clusters = (list(gen_resource_clusters()) if not multi_worker else
                         list(gen_multi_worker_resource_clusters()))

    weights = np.ones((len(workers), len(resource_clusters))) * -9999
    for j, cid in enumerate(resource_clusters):
      for i, worker in enumerate(workers):
        weights[i, j] = get_cluster_weight(worker, cid)


    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    sorted_pairs = sorted(list(zip(rows, cols)), key=lambda x: -weights[x[0], x[1]])
    for worker_idx, cluster_idx in sorted_pairs:
      worker = workers[worker_idx]
      cid = resource_clusters[cluster_idx]

      # tile_pos = worker.cid_to_tile_pos[cid]
      # print(f'Assign Cluster {worker.id}, cell[{tile_pos}], wt={weights[worker_idx, cluster_idx]}')
      if weights[worker_idx, cluster_idx] < 0:
        continue

      # Keep the mapping, but not do cluster boost (not good)
      # n_player_citytile = self.cluster_info.player_citytile_count[cid]
      # if n_player_citytile > 0:
        # continue

      worker.target_cluster_id = cid
      worker.target_cid_turns = worker.cid_to_cluster_turns[cid]
      worker.is_cluster_owner = True

      if DRAW_UNIT_CLUSTER_PAIR:
        tile_pos = worker.cid_to_tile_pos[cid]
        x = annotate.x(tile_pos.x, tile_pos.y)
        line = annotate.line(worker.pos.x, worker.pos.y, tile_pos.x, tile_pos.y)
        self.actions.extend([x, line])

        # print(f'Assign Cluster {worker.id}, cell[{tile_pos}], wt={weights[worker_idx, cluster_idx]}')



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
        elif is_resource_researched(nb_cell.resource, self.player):
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
        # if worker.id in DRAW_UNIT_LIST:
          # print(f" c={target_cell.pos}, wt={get_neighbour_cell_weight(target_cell)}")
        # Target cell must can build citytile
        if not cell_can_build_citytile(target_cell):
          continue

        # TODO: try make one step collections value more accurate
        # Worker can collect on its own.
        collect_amt, _ = get_one_step_collection_values(target_cell, self.game.player, self.game)
        # collect_amt = 0
        if worker_amt + collect_amt >= CITY_BUILD_COST:
        # collect_amt /= 2  # Discount this value
        # if worker_amt + collect_amt >= CITY_BUILD_COST:
        # if worker_amt >= CITY_BUILD_COST:
          break

        # transfer + worker.action + worker.collect on next step
        for nb_cell in nb_cells:
          nb_unit = nb_cell.unit
          debug = False
          # if nb_unit and nb_unit.id == 'u_4' and worker.id == 'u_6':
            # debug = True
            # print(f" nb_unit {nb_unit.id}, left={nb_unit.get_cargo_space_left()} has_planned_action={nb_unit.has_planned_action}, can_act={nb_unit.can_act()}")

          # Neighbour is not valid or ready.
          if (nb_unit is None
              or nb_unit.team != self.game.player.team
              or not nb_unit.can_act()
              or nb_unit.has_planned_action):
            continue

          # Neighbour can build on its own cell.
          is_nb_unit_can_build = cell_can_build_citytile(nb_cell)
          is_nb_unit_can_move_and_build = (neighbour_cells_can_build_citytile(nb_cell)
                                           and (self.game.turn + 3) % CIRCLE_LENGH < BUILD_CITYTILE_ROUND)
          if (nb_unit.get_cargo_space_left() == 0
              and (is_nb_unit_can_build or is_nb_unit_can_move_and_build)):
            # if debug:
              # print(f' nb_can_build={is_nb_unit_can_build}, move_and_build={is_nb_unit_can_move_and_build}')
            continue


          # TODO: support other resource: use max resource
          if (worker_amt + collect_amt + nb_unit.cargo.wood >= CITY_BUILD_COST):
            transfer_amount = CITY_BUILD_COST - (worker_amt + collect_amt)
            # transfer_amount = CITY_BUILD_COST - (worker_amt)
            # transfer_amount = nb_unit.cargo.wood
            print(f'$A {worker.id}{worker.cargo}@{worker.pos} accept transfer from {nb_unit.id}{nb_unit.cargo} ${transfer_amount} to goto {target_cell.pos}')
            self.add_unit_action(nb_unit,
                                 nb_unit.transfer(worker.id, RESOURCE_TYPES.WOOD, transfer_amount))
            nb_unit.is_transfer_worker = True
            worker.transfer_build_locations.add(target_cell.pos)

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
      # print(f'{worker.id} is sending to {city.cityid}')

  def is_worker_building_citytile(self, worker, target_pos):
    return (worker.id, target_pos) in self.worker_build_city_tasks

  def is_worker_fuel_city_task(self, worker, target_pos):
    return (worker.id, target_pos) in self.worker_fuel_city_tasks

  def execute(self):
    actions = self.actions
    g = self.game
    player = g.player

    # MAIN_PRINT
    print((f'>> turn={g.turn}, #W={len(player.units)}, #C={player.city_tile_count} '
           f'R={player.research_points}'), file=sys.stderr)

    self.update_game_info()

    self.compute_citytile_actions()

    # self.assign_worker_city_target()
    self.worker_look_for_resource_transfer()

    self.assign_worker_to_resource_cluster()
    # self.assign_worker_to_resource_cluster(multi_worker=True)
    self.assign_worker_target()

    self.try_build_citytile()
    self.compute_worker_moves()


_strategy = Strategy()

def agent(observation, configuration):
  _strategy.update(observation, configuration)
  _strategy.execute()
  return _strategy.actions
