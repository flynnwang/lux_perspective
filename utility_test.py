from utility3 import *
from lux.game_objects import Cargo, City, Unit


def _make_cargo(w, c, u):
  cargo = Cargo()
  cargo.wood = w
  cargo.coal = c
  cargo.uranium = u
  return cargo


def test_cargo_night_endurance():
  upkeep = 4
  cargo = _make_cargo(4, 0, 0)
  assert cargo_night_endurance(cargo, upkeep) == 1

  cargo = _make_cargo(5, 0, 0)
  assert cargo_night_endurance(cargo, upkeep) == 1

  cargo = _make_cargo(9, 0, 0)
  assert cargo_night_endurance(cargo, upkeep) == 2

  cargo = _make_cargo(9, 1, 0)
  assert cargo_night_endurance(cargo, upkeep) == 3

  cargo = _make_cargo(9, 0, 1)
  assert cargo_night_endurance(cargo, upkeep) == 3

  cargo = _make_cargo(9, 1, 1)
  assert cargo_night_endurance(cargo, upkeep) == 4

  cargo = _make_cargo(0, 1, 0)
  assert cargo_night_endurance(cargo, upkeep) == 1

  cargo = _make_cargo(0, 0, 1)
  assert cargo_night_endurance(cargo, upkeep) == 1


test_cargo_night_endurance()


def test_get_night_count_by_days():
  assert get_night_count_by_days(0, 3) == 0
  assert get_night_count_by_days(0, 30) == 0

  assert get_night_count_by_days(0, 31) == 1
  assert get_night_count_by_days(0, 39) == 9
  assert get_night_count_by_days(0, 40) == 10

  assert get_night_count_by_days(0, 41) == 10
  assert get_night_count_by_days(0, 51) == 10
  assert get_night_count_by_days(0, 70) == 10
  assert get_night_count_by_days(0, 75) == 15
  assert get_night_count_by_days(0, 80) == 20

  assert get_night_count_by_days(10, 3) == 0
  assert get_night_count_by_days(10, 19) == 0
  assert get_night_count_by_days(10, 20) == 0
  assert get_night_count_by_days(10, 21) == 1
  assert get_night_count_by_days(10, 30) == 10


def test_get_night_count_by_dist():
  cooldown = 2

  assert get_night_count_by_dist(0, 1, 0, cooldown) == (0, 1)
  assert get_night_count_by_dist(0, 5, 0, cooldown) == (0, 9)
  assert get_night_count_by_dist(0, 15, 0, cooldown) == (0, 29)

  # # # dist=15 for 30 days, dist 2 for 8 night days
  # # # one night cost double cooldown
  assert get_night_count_by_dist(0, 16, 0, cooldown) == (1, 30 + 1)
  assert get_night_count_by_dist(0, 17, 0, cooldown) == (5, 30 + 5)
  assert get_night_count_by_dist(0, 18, 0, cooldown) == (9, 30 + 9)
  assert get_night_count_by_dist(0, 19, 0, cooldown) == (10, 30 + 4 + 4 + 4 + 1)

  # # # test unit_cooldown
  assert get_night_count_by_dist(0, 15, 1, cooldown) == (0, 29 + 1)
  assert get_night_count_by_dist(0, 15, 2, cooldown) == (1, 2 + 14 * 2 + 1)
  assert get_night_count_by_dist(0, 15, 3, cooldown) == (2, 3 + 13 * 2 + 2 + 1)
  assert get_night_count_by_dist(0, 15, 4, cooldown) == (5, 4 + 13 * 2 + 4 + 1)

  # # test initial turn
  # assert get_night_count_by_dist(1, 15, 0, cooldown) == 0
  # assert get_night_count_by_dist(2, 15, 0, cooldown) == 1
  # assert get_night_count_by_dist(3, 15, 0, cooldown) == 2
  # assert get_night_count_by_dist(4, 15, 0, cooldown) == 5


def test_get_city_no():
  city = City(0, 'c_10', 0, 0)

  assert get_city_no(city) == 10


def test_constants():
  assert MAX_RESEARCH_POINTS == 200


def test_city_wont_last_at_nights():
  city = City(0, 'c_10', fuel=4, light_upkeep=2)
  assert not city_wont_last_at_nights(39, city)

  city = City(0, 'c_10', fuel=4, light_upkeep=2)
  assert not city_wont_last_at_nights(38, city)

  city = City(0, 'c_10', fuel=4, light_upkeep=2)
  assert city_wont_last_at_nights(37, city)

  city = City(0, 'c_10', fuel=4, light_upkeep=2)
  assert city_wont_last_at_nights(38, city) == False

  # city = City(0, 'c_10', fuel=5, light_upkeep=2)
  # assert not city_wont_last_at_nights(38, city)


def test_get_night_count_this_round():
  assert get_night_count_this_round(69) == 10

  assert get_night_count_this_round(0) == 10
  assert get_night_count_this_round(29) == 10
  assert get_night_count_this_round(30) == 10
  assert get_night_count_this_round(39) == 1
  assert get_night_count_this_round(40) == 10


def test_compute_last_turns():
  assert nights_to_last_turns(0, 5) == 30 + 5
  assert nights_to_last_turns(0, 2) == 30 + 2
  assert nights_to_last_turns(0, 1) == 30 + 1
  assert nights_to_last_turns(0, 0) == 30

  assert nights_to_last_turns(30, 5) == 5
  assert nights_to_last_turns(30, 2) == 2
  assert nights_to_last_turns(30, 1) == 1
  assert nights_to_last_turns(30, 0) == 0

  assert nights_to_last_turns(31, 5) == 5
  assert nights_to_last_turns(31, 2) == 2
  assert nights_to_last_turns(31, 1) == 1
  assert nights_to_last_turns(31, 0) == 0

  assert nights_to_last_turns(29, 5) == 5 + 1
  assert nights_to_last_turns(29, 2) == 2 + 1
  assert nights_to_last_turns(29, 1) == 1 + 1
  assert nights_to_last_turns(29, 0) == 0 + 1

  assert nights_to_last_turns(0, 10) == 70
  assert nights_to_last_turns(1, 10) == 69
  assert nights_to_last_turns(213, 40) == 147


def test_consume_cargo_at_citytile():
  turn = 0
  cargo = Cargo(100, 0, 0)
  upkeep = WORKER_UPKEEP
  assert consume_cargo(turn, cargo, True, 1, upkeep) == Cargo()
  assert consume_cargo(turn, cargo, True, 10, upkeep) == Cargo()
  assert consume_cargo(turn, cargo, True, 40, upkeep) == Cargo()

  assert consume_cargo(turn, cargo, True, 0, upkeep) == Cargo()


def test_consume_cargo_in_the_day():
  cargo = Cargo(100, 0, 0)
  upkeep = WORKER_UPKEEP
  assert consume_cargo(0, cargo, False, 1, upkeep) == cargo
  assert consume_cargo(10, cargo, False, 1, upkeep) == cargo
  assert consume_cargo(20, cargo, False, 1, upkeep) == cargo


def test_consume_cargo_at_night():
  cargo = Cargo(100, 0, 0)
  upkeep = WORKER_UPKEEP
  print(str(cargo))
  assert consume_cargo(30, cargo, False, 1, upkeep) == Cargo(96, 0, 0)
  assert consume_cargo(30, cargo, False, 2, upkeep) == Cargo(92, 0, 0)

  # night + day
  assert consume_cargo(38, cargo, False, 2, upkeep) == Cargo(96, 0, 0)


def test_is_day_and_night():
  assert is_day(0)
  assert is_day(29)
  assert is_night(30)
  assert is_night(39)


def test_unit_arrival_turns():
  teamid = 0
  unitid = 'c_1'
  unit = Unit(teamid,
              Constants.UNIT_TYPES.WORKER,
              unitid,
              x=0,
              y=0,
              cooldown=1,
              wood=0,
              coal=0,
              uranium=0)

  turn = 13
  dist = 5
  action_cooldown = get_unit_action_cooldown(unit)
  assert action_cooldown == 2

  nights, days = get_night_count_by_dist(turn, dist, unit.cooldown,
                                         action_cooldown)
  assert nights == 0
  assert days == 10

  assert unit_arrival_turns(turn, unit, dist) == 10


def test_remaining_round():

  assert get_remaining_round(0) == 8
  assert get_remaining_round(1) == 8
  assert get_remaining_round(39) == 8
  assert get_remaining_round(40) == 7
  assert get_remaining_round(79) == 7
  assert get_remaining_round(80) == 6


def test_remaining_nights():

  assert get_remaining_nights(0) == 9 * 10
  assert get_remaining_nights(1) == 9 * 10

  assert get_remaining_nights(29) == 9 * 10
  assert get_remaining_nights(30) == 9 * 10

  assert get_remaining_nights(39) == 8 * 10 + 1
  assert get_remaining_nights(40) == 8 * 10


def test_day_count_by_arrival_turns():

  assert day_count_by_arrival_turns(0, 10) == 10
  assert day_count_by_arrival_turns(0, 30) == 30
  assert day_count_by_arrival_turns(0, 35) == 30
  assert day_count_by_arrival_turns(0, 40) == 30
  assert day_count_by_arrival_turns(0, 41) == 31
