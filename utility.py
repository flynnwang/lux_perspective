
import math
from copy import deepcopy

from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux.game_map import Resource


def params(key):
  return GAME_CONSTANTS['PARAMETERS'][key]


DIRECTIONS = Constants.DIRECTIONS
DAY_LENGTH = params('DAY_LENGTH')
NIGHT_LENGTH = params('NIGHT_LENGTH')
CIRCLE_LENGH = DAY_LENGTH + NIGHT_LENGTH

UNIT_ACTION_COOLDOWN = params('UNIT_ACTION_COOLDOWN')
WORKER_ACTION_COOLDOWN = UNIT_ACTION_COOLDOWN["WORKER"]
CITY_BUILD_COST = params('CITY_BUILD_COST')

LIGHT_UPKEEP = params('LIGHT_UPKEEP')

MAX_RESEARCH_POINTS = params("RESEARCH_REQUIREMENTS")["URANIUM"]

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


def cargo_night_endurance(cargo, upkeep):
  cargo = deepcopy(cargo)
  if cargo.wood == 0 and cargo.coal == 0 and cargo.uranium == 0:
    return 0

  wood_fuel_rate = get_resource_to_fuel_rate('WOOD')
  coal_fuel_rate = get_resource_to_fuel_rate('COAL')
  uranium_fuel_rate = get_resource_to_fuel_rate('URANIUM')

  def burn_fuel(amount, fuel_rate):
    one_night_amount = int(math.ceil(upkeep / fuel_rate))
    nights = amount // one_night_amount
    resource_left = amount - one_night_amount * nights
    return nights, resource_left

  wood_nights, wood_left = burn_fuel(cargo.wood, wood_fuel_rate)

  assert coal_fuel_rate >= upkeep and uranium_fuel_rate >= upkeep
  if wood_left > 0 and (cargo.coal > 0 or cargo.uranium > 0):
    wood_nights += 1

    if cargo.coal > 0:
      cargo.coal -= 1
    else:
      assert cargo.uranium > 0
      cargo.uranium -= 1

  coal_nights, _ = burn_fuel(cargo.coal, coal_fuel_rate)
  uranium_nights, _ = burn_fuel(cargo.uranium, uranium_fuel_rate)
  return wood_nights + coal_nights + uranium_nights


def is_day(turn):
  turn %= CIRCLE_LENGH
  return turn < DAY_LENGTH


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
    return nights + + nights_left + get_night_count_by_dist(turn, next_dist,
                                                            unit_cooldown, cooldown)


def city_wont_last_at_nights(turn, city):
  fuel = city.fuel
  light_upkeep = city.light_upkeep

  # TODO(): can add more value to make city last
  turn %= CIRCLE_LENGH
  nights = min(CIRCLE_LENGH - turn, NIGHT_LENGTH)
  return fuel // light_upkeep < nights



