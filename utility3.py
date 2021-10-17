import functools
import math
from copy import deepcopy

from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux.game_map import Resource
from lux.game_objects import Cargo


def params(key):
  return GAME_CONSTANTS['PARAMETERS'][key]


MAX_DAYS = params("MAX_DAYS")

DIRECTIONS = Constants.DIRECTIONS
DAY_LENGTH = params('DAY_LENGTH')
NIGHT_LENGTH = params('NIGHT_LENGTH')
CIRCLE_LENGH = DAY_LENGTH + NIGHT_LENGTH

UNIT_ACTION_COOLDOWN = params('UNIT_ACTION_COOLDOWN')
WORKER_ACTION_COOLDOWN = UNIT_ACTION_COOLDOWN["WORKER"]
CART_ACTION_COOLDOWN = UNIT_ACTION_COOLDOWN["CART"]

WORKER_RESOURCE_CAPACITY = params('RESOURCE_CAPACITY')['WORKER']
CART_RESOURCE_CAPACITY = params('RESOURCE_CAPACITY')['CART']

WORKER_COLLECTION_RATE = params('WORKER_COLLECTION_RATE')
CITY_BUILD_COST = params('CITY_BUILD_COST')
CITY_ACTION_COOLDOWN = params('CITY_ACTION_COOLDOWN')

LIGHT_UPKEEP = params('LIGHT_UPKEEP')
WORKER_UPKEEP = LIGHT_UPKEEP['WORKER']
CART_UPKEEP = LIGHT_UPKEEP['CART']

MAX_RESEARCH_POINTS = params("RESEARCH_REQUIREMENTS")["URANIUM"]  # old version
URANIUM_RESEARCH_POINTS = params("RESEARCH_REQUIREMENTS")["URANIUM"]
COAL_RESEARCH_POINTS = params("RESEARCH_REQUIREMENTS")["COAL"]

MAX_WOOD_AMOUNT = params("MAX_WOOD_AMOUNT")

# ORDER MATTERS
ALL_RESOURCE_TYPES = ["URANIUM", "COAL", "WOOD"]


def is_within_map_range(pos, game_map):
  return 0 <= pos.x < game_map.width and 0 <= pos.y < game_map.height


def get_unit_upkeep(unit):
  if unit.is_worker():
    return WORKER_UPKEEP
  assert unit.is_cart()
  return CART_UPKEEP


# Deprecated
def get_unit_upkeep_by_type(unit_type):
  if unit_type == Constants.UNIT_TYPES.WORKER:
    return WORKER_UPKEEP
  assert unit_type == Constants.UNIT_TYPES.CART
  return CART_UPKEEP


def get_unit_capacity_by_type(unit_type):
  if unit_type == Constants.UNIT_TYPES.WORKER:
    return WORKER_RESOURCE_CAPACITY
  assert unit_type == Constants.UNIT_TYPES.CART
  return CART_RESOURCE_CAPACITY


def get_unit_action_cooldown(unit):
  if unit.is_worker():
    return WORKER_ACTION_COOLDOWN
  assert unit.is_cart()
  return CART_ACTION_COOLDOWN


def get_city_no(city):
  return int(city.cityid.split('_')[1])


def get_resource_to_fuel_rate(resource):
  type_name = resource
  if isinstance(resource, Resource):
    type_name = resource.type.upper()
  return params('RESOURCE_TO_FUEL_RATE')[type_name]


WOOD_FUEL_RATE = get_resource_to_fuel_rate('WOOD')
COAL_FUEL_RATE = get_resource_to_fuel_rate('COAL')
URANIUM_FUEL_RATE = get_resource_to_fuel_rate('URANIUM')


def get_worker_collection_rate(resource):
  type_name = resource
  if isinstance(resource, Resource):
    type_name = resource.type.upper()
  return params('WORKER_COLLECTION_RATE')[type_name]


def cargo_total_amount(cargo):
  return cargo.wood + cargo.coal + cargo.uranium
  # v = getattr(cargo, 'total_amount', None)
  # if v is None:
  # v = cargo.wood + cargo.coal + cargo.uranium
  # cargo.total_amount = v
  # return v


def add_resource_to_cargo(cargo, capacity, amt, res_type):
  total_amount = cargo_total_amount(cargo)
  assert total_amount <= capacity, f"amt={amt}, total={total_amount}"
  amt = min(capacity - total_amount, amt)
  if res_type == 'WOOD':
    cargo.wood += amt
  elif res_type == 'COAL':
    cargo.coal += amt
  else:
    assert res_type == 'URANIUM'
    cargo.uranium += amt


def worker_total_cargo(worker):
  cargo = worker.cargo
  return cargo_total_amount(cargo)


def is_worker_cargo_full(worker):
  return worker_total_cargo(worker) == WORKER_RESOURCE_CAPACITY


def cargo_total_fuel(cargo):
  return (cargo.wood * WOOD_FUEL_RATE + cargo.coal * COAL_FUEL_RATE +
          cargo.uranium * URANIUM_FUEL_RATE)
  # Cache the fuel on cargo
  # v = getattr(cargo, 'fuel', None)
  # if v is None:
  # v = (cargo.wood * WOOD_FUEL_RATE
  # + cargo.coal * COAL_FUEL_RATE
  # + cargo.uranium * URANIUM_FUEL_RATE)
  # cargo.fuel = v
  return v


def worker_total_fuel(worker):
  return cargo_total_fuel(worker.cargo)


def worker_cargo_full_rate(worker):
  return worker_total_cargo(worker) / WORKER_RESOURCE_CAPACITY


def is_resource_wood(resource):
  return (resource.type == Constants.RESOURCE_TYPES.WOOD and
          resource.amount > 0)


def is_resource_coal(resource):
  return (resource.type == Constants.RESOURCE_TYPES.COAL and
          resource.amount > 0)


def is_resource_uranium(resource):
  return (resource.type == Constants.RESOURCE_TYPES.URANIUM and
          resource.amount > 0)


def cell_has_opponent_citytile(cell, game):
  citytile = cell.citytile
  return citytile is not None and citytile.team == game.opponent.team


def cell_has_player_citytile(cell, game):
  citytile = cell.citytile
  return citytile is not None and citytile.team == game.player.team


def cell_has_target_player_citytile(cell, player):
  citytile = cell.citytile
  return citytile is not None and citytile.team == player.team


def is_last_night(turn):
  return (turn % CIRCLE_LENGH) == (CIRCLE_LENGH - 1)


def resource_to_cargo(resource):
  cargo = Cargo()
  if resource is None:
    return cargo
  if is_resource_wood(resource):
    cargo.wood = resource.amount
  if is_resource_coal(resource):
    cargo.coal = resource.amount
  if is_resource_uranium(resource):
    cargo.uranium = resource.amount
  return cargo


def resource_fuel(resource):
  return resource.amount * get_resource_to_fuel_rate(resource)


def cargo_night_endurance(cargo, upkeep):
  cargo = deepcopy(cargo)
  if cargo.wood == 0 and cargo.coal == 0 and cargo.uranium == 0:
    return 0

  def burn_fuel(amount, fuel_rate):
    one_night_amount = int(math.ceil(upkeep / fuel_rate))
    nights = amount // one_night_amount
    resource_left = amount - one_night_amount * nights
    return nights, resource_left

  wood_nights, wood_left = burn_fuel(cargo.wood, WOOD_FUEL_RATE)

  assert COAL_FUEL_RATE >= upkeep and URANIUM_FUEL_RATE >= upkeep
  if wood_left > 0 and (cargo.coal > 0 or cargo.uranium > 0):
    wood_nights += 1

    if cargo.coal > 0:
      cargo.coal -= 1
    else:
      assert cargo.uranium > 0
      cargo.uranium -= 1

  coal_nights, _ = burn_fuel(cargo.coal, COAL_FUEL_RATE)
  uranium_nights, _ = burn_fuel(cargo.uranium, URANIUM_FUEL_RATE)
  return wood_nights + coal_nights + uranium_nights


def consume_cargo(turn, cargo, is_citytile, sim_turns, upkeep):
  """1) Not counting resource collect on the way.
     2) Not counting city corrupt."""
  # Empty cargo when worker is on citytile.
  if is_citytile:
    return Cargo()

  if sim_turns == 0:
    return cargo

  def burn_resource(resource_amt, fuel_rate, burn):
    if burn == 0:
      return resource_amt, 0

    one_night_amount = int(math.ceil(upkeep / fuel_rate))
    if resource_amt >= one_night_amount:
      return resource_amt - one_night_amount, 0
    return 0, burn - resource_amt * fuel_rate

  cargo = Cargo(cargo.wood, cargo.coal, cargo.uranium)
  # for t in range(turn+1, turn+sim_turns+1):
  for t in range(turn + 1, turn + sim_turns + 1):
    # Cargo won't change during the day.
    if is_day(t):
      continue

    burn = upkeep
    cargo.wood, burn = burn_resource(cargo.wood, WOOD_FUEL_RATE, burn)
    cargo.coal, burn = burn_resource(cargo.coal, COAL_FUEL_RATE, burn)
    cargo.uranium, burn = burn_resource(cargo.uranium, URANIUM_FUEL_RATE, burn)

    # No enough resource to fullfill upkeep
    if burn > 0:
      return None
  return cargo


@functools.lru_cache(maxsize=MAX_DAYS)
def is_day(turn):
  turn %= CIRCLE_LENGH
  return turn < DAY_LENGTH


@functools.lru_cache(maxsize=MAX_DAYS)
def is_night(turn):
  return not is_day(turn)


def get_night_count_this_round(turn):
  turn %= CIRCLE_LENGH
  return min(CIRCLE_LENGH - turn, NIGHT_LENGTH)


def is_first_night(turn):
  turn %= CIRCLE_LENGH
  return turn == DAY_LENGTH


def get_day_count_this_round(turn):
  turn %= CIRCLE_LENGH
  return max(DAY_LENGTH - turn, 0)


def get_left_turns_this_round(turn):
  return CIRCLE_LENGH - turn % CIRCLE_LENGH


def get_night_count_by_days(turn, days):
  turn %= CIRCLE_LENGH
  if is_day(turn):
    days_left = DAY_LENGTH - turn
    if days_left >= days:
      return 0

    days -= days_left
    circle_count = days // CIRCLE_LENGH
    remain_days = days - circle_count * CIRCLE_LENGH
    last_nights = min(remain_days, NIGHT_LENGTH)
    return circle_count * NIGHT_LENGTH + last_nights
  else:
    nights_left = CIRCLE_LENGH - turn
    if nights_left > days:
      return days

    days -= nights_left
    circle_count = days // CIRCLE_LENGH
    remain_days = days - circle_count * CIRCLE_LENGH
    last_nights = max(0, remain_days - DAY_LENGTH)
    return circle_count * NIGHT_LENGTH + last_nights


def get_night_count_by_dist(turn, dist, unit_cooldown, cooldown):
  if dist <= 0:
    return 0, 0

  def estimate_days(d, c):
    return (d - 1) * c + 1

  # Initial cooldown
  total_turns = unit_cooldown
  nights = get_night_count_by_days(turn, unit_cooldown)
  turn += unit_cooldown
  turn %= CIRCLE_LENGH
  if is_day(turn):
    days_left = DAY_LENGTH - turn
    move_days = estimate_days(dist, cooldown)
    # print(f'days_left = {days_left}, move_days={move_days}')
    if days_left >= move_days:
      return nights, total_turns + move_days

    turn += days_left
    step = int(math.ceil(days_left / cooldown))
    next_dist = dist - step
    # print(f'turn={turn}, next_dist={next_dist}')
    assert step > 0
    unit_cooldown = step * cooldown - days_left
    other_night_turns, other_day_turns = get_night_count_by_dist(
        turn, next_dist, unit_cooldown, cooldown)
    # print(f'nights={nights}, remain={other}')
    return (nights + other_night_turns,
            total_turns + days_left + other_day_turns)
  else:
    night_cooldown = cooldown * 2  # double at nihght
    nights_left = CIRCLE_LENGH - turn
    night_travel_days = estimate_days(dist, night_cooldown)
    if nights_left >= night_travel_days:
      # print(f'night-1: nights={nights}, night_travel_days={night_travel_days}')
      return nights + night_travel_days, total_turns + night_travel_days

    turn += nights_left
    step = int(math.ceil(nights_left / night_cooldown))
    assert step > 0
    next_dist = dist - step
    unit_cooldown = step * night_cooldown - nights_left
    # print(f'night-2, nights_left={nights_left}, step={step}, unit_cooldown={unit_cooldown}')
    other_night_turns, other_day_turns = get_night_count_by_dist(
        turn, next_dist, unit_cooldown, cooldown)
    return (nights + nights_left + other_night_turns,
            total_turns + nights_left + other_day_turns)


def city_last_nights(city, add_fuel=0):
  fuel = city.fuel + add_fuel
  light_upkeep = city.light_upkeep
  return fuel // light_upkeep + 1


def nights_to_last_turns(turn, last_nights):
  if turn >= MAX_DAYS:
    return 0

  if is_day(turn):
    days = get_day_count_this_round(turn)
    # print(f'add days={days}')
    return nights_to_last_turns(turn + days, last_nights) + days

  # The night case
  nights = get_night_count_this_round(turn)
  if last_nights < nights:
    return last_nights

  last_nights -= nights
  # print(f'add nights={days}')
  return nights_to_last_turns(turn + nights, last_nights) + nights


def city_last_days(turn, city):
  last_nights = city_last_nights(city)
  return nights_to_last_turns(turn, last_nights)


def unit_surviving_turns(turn, unit):
  last_nights = cargo_night_endurance(unit.cargo, get_unit_upkeep(unit))
  return nights_to_last_turns(turn, last_nights)


def unit_arrival_turns(turn, unit, dist):
  cooldown = get_unit_action_cooldown(unit)
  _, turns = get_night_count_by_dist(turn, dist, unit.cooldown, cooldown)
  return turns


def resource_surviving_nights(turn, resource, upkeep):
  cargo = resource_to_cargo(resource)
  nights = cargo_night_endurance(cargo, upkeep)
  return nights_to_last_turns(turn, nights)


def city_wont_last_at_nights(turn, city, add_fuel=0):
  round_nights = get_night_count_this_round(turn)
  city_nights = city_last_nights(city, add_fuel)
  return city_nights < round_nights


def get_remaining_round(turn):
  """Not including this round."""
  return (MAX_DAYS - turn - 1) // CIRCLE_LENGH


def get_remaining_nights(turn):
  """nights remaining, including current turn."""
  nights = get_night_count_this_round(turn)
  round = get_remaining_round(turn)
  return nights + round * NIGHT_LENGTH
