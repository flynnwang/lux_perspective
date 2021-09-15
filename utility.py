
import math
from copy import deepcopy

from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux.game_map import Resource
from lux.game_objects import Cargo


def params(key):
  return GAME_CONSTANTS['PARAMETERS'][key]


DIRECTIONS = Constants.DIRECTIONS
DAY_LENGTH = params('DAY_LENGTH')
NIGHT_LENGTH = params('NIGHT_LENGTH')
CIRCLE_LENGH = DAY_LENGTH + NIGHT_LENGTH

UNIT_ACTION_COOLDOWN = params('UNIT_ACTION_COOLDOWN')
WORKER_ACTION_COOLDOWN = UNIT_ACTION_COOLDOWN["WORKER"]
WORKER_RESOURCE_CAPACITY = params('RESOURCE_CAPACITY')['WORKER']
CITY_BUILD_COST = params('CITY_BUILD_COST')
CITY_ACTION_COOLDOWN = params('CITY_ACTION_COOLDOWN')

LIGHT_UPKEEP = params('LIGHT_UPKEEP')
WORKER_UPKEEP = LIGHT_UPKEEP['WORKER']

MAX_RESEARCH_POINTS = params("RESEARCH_REQUIREMENTS")["URANIUM"]




def is_within_map_range(pos, game_map):
  return 0 <= pos.x < game_map.width and 0 <= pos.y < game_map.height


def get_unit_type_string(unit_type):
  return 'WORKER' if unit_type == Constants.UNIT_TYPES.WORKER else 'CART'

def get_unit_upkeep(unit):
  return LIGHT_UPKEEP[get_unit_type_string(unit.type)]

def get_unit_action_cooldown(unit):
  return UNIT_ACTION_COOLDOWN[get_unit_type_string(unit.type)]


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


def worker_total_cargo(worker):
  cargo = worker.cargo
  return (cargo.wood + cargo.coal + cargo.uranium)


def cargo_total_fuel(cargo):
  return (cargo.wood * WOOD_FUEL_RATE
          + cargo.coal * COAL_FUEL_RATE
          + cargo.uranium * URANIUM_FUEL_RATE)


def worker_total_fuel(worker):
  return cargo_total_fuel(worker.cargo)


def worker_cargo_full_rate(worker):
  return worker_total_cargo(worker) / WORKER_RESOURCE_CAPACITY


def is_resource_wood(resource):
  return (resource.type == Constants.RESOURCE_TYPES.WOOD
          and resource.amount > 0)

def is_resource_coal(resource):
  return (resource.type == Constants.RESOURCE_TYPES.COAL
          and resource.amount > 0)

def is_resource_uranium(resource):
  return (resource.type == Constants.RESOURCE_TYPES.URANIUM
          and resource.amount > 0)


def cell_has_opponent_citytile(cell, game):
  citytile = cell.citytile
  return citytile is not None and citytile.team == game.opponent.team

def cell_has_player_citytile(cell, game):
  citytile = cell.citytile
  return citytile is not None and citytile.team == game.player.team



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
  # Empty cargo when worker is on citytile.
  if is_citytile:
    return Cargo()

  if sim_turns == 0:
    return cargo

  def burn_resource(resource_amt, fuel_rate, fuel):
    if fuel == 0:
      return resource_amt, 0

    one_night_amount = int(math.ceil(upkeep / fuel_rate))
    if resource_amt >= one_night_amount:
      return resource_amt - one_night_amount, 0
    return 0, fuel - resource_amt * fuel_rate

  cargo = Cargo(cargo.wood, cargo.coal, cargo.uranium)
  # for t in range(turn+1, turn+sim_turns+1):
  for t in range(turn+1, turn+sim_turns+1):
    # Cargo won't change during the day.
    if is_day(t):
      continue

    fuel = upkeep
    cargo.wood, fuel = burn_resource(cargo.wood, WOOD_FUEL_RATE, fuel)
    cargo.coal, fuel = burn_resource(cargo.coal, COAL_FUEL_RATE, fuel)
    cargo.uranium, fuel = burn_resource(cargo.uranium, URANIUM_FUEL_RATE, fuel)

    # No enough resource to collect fuel
    if fuel > 0:
      return None
  return cargo


def is_day(turn):
  turn %= CIRCLE_LENGH
  return turn < DAY_LENGTH


def is_night(turn):
  return not is_day(turn)


def get_night_count_this_round(turn):
  turn %= CIRCLE_LENGH
  return min(CIRCLE_LENGH - turn, NIGHT_LENGTH)


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
    return 0

  def estimate_days(d, c):
    return (d - 1) * c + 1

  # Initial cooldown
  nights = get_night_count_by_days(turn, unit_cooldown)
  turn += unit_cooldown
  turn %= CIRCLE_LENGH
  if is_day(turn):
    days_left = DAY_LENGTH - turn
    # print(f'days_left = {days_left}, {dist * cooldown}')
    if days_left >= estimate_days(dist, cooldown):
      return nights

    turn += days_left
    step = int(math.ceil(days_left / cooldown))
    # print(f'turn={turn}, step={step}, dist={dist}')
    # dist -= step
    next_dist = dist - step
    assert step > 0
    unit_cooldown = step * cooldown - days_left
    other = get_night_count_by_dist(turn, next_dist, unit_cooldown, cooldown)
    # print(f'nights={nights}, remain={other}')
    return nights + other
  else:
    night_cooldown = cooldown * 2 # double at nihght
    nights_left = CIRCLE_LENGH - turn
    night_travel_days = estimate_days(dist, night_cooldown)
    if nights_left >= night_travel_days:
      # print(f'night-1: nights={nights}, night_travel_days={night_travel_days}')
      return nights + night_travel_days

    turn += nights_left
    step = int(math.ceil(nights_left / night_cooldown))
    assert step > 0
    next_dist = dist - step
    unit_cooldown = step * night_cooldown - nights_left
    # print(f'night-2, nights_left={nights_left}, step={step}, unit_cooldown={unit_cooldown}')
    return nights + nights_left + get_night_count_by_dist(turn, next_dist,
                                                           unit_cooldown, cooldown)


def city_last_nights(city, add_fuel=0):
  fuel = city.fuel + add_fuel
  light_upkeep = city.light_upkeep
  return fuel // light_upkeep


def nights_to_last_turns(turn, last_nights):
  if is_day(turn):
    days = get_day_count_this_round(turn)
    return nights_to_last_turns(turn+days, last_nights) + days

  # The night case
  nights = get_night_count_this_round(turn)
  if last_nights <= nights:
    return last_nights

  last_nights -= nights
  return nights_to_last_turns(turn+nights, last_nights)


def city_last_days(turn, city):
  last_nights = city_last_nights(city)
  return nights_to_last_turns(turn, last_nights)


def unit_surviving_turns(turn, unit):
  last_nights = cargo_night_endurance(unit.cargo, get_unit_upkeep(unit))
  return nights_to_last_turns(turn, last_nights)


def unit_arrival_turns(turn, unit, dist):
  cooldown = get_unit_action_cooldown(unit)
  last_nights = get_night_count_by_dist(turn, dist, unit.cooldown, cooldown)
  return nights_to_last_turns(turn, last_nights)


def resource_surviving_nights(turn, resource, upkeep):
  cargo = resource_to_cargo(resource)
  nights = cargo_night_endurance(cargo, upkeep)
  return nights_to_last_turns(turn, nights)



def city_wont_last_at_nights(turn, city, add_fuel=0):
  turn %= CIRCLE_LENGH
  round_nights = min(CIRCLE_LENGH - turn, NIGHT_LENGTH)
  city_nights = city_last_nights(city, add_fuel)
  return city_nights < round_nights



