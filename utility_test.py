

from utility import *
from lux.game_objects import Cargo, City


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

  assert get_night_count_by_dist(0, 1, 0, cooldown) == 0
  assert get_night_count_by_dist(0, 5, 0, cooldown) == 0
  assert get_night_count_by_dist(0, 15, 0, cooldown) == 0

  # dist=15 for 30 days, dist 2 for 8 night days
  # one night cost double cooldown
  assert get_night_count_by_dist(0, 16, 0, cooldown) == 1
  assert get_night_count_by_dist(0, 17, 0, cooldown) == 5
  assert get_night_count_by_dist(0, 18, 0, cooldown) == 9
  assert get_night_count_by_dist(0, 19, 0, cooldown) == 10

  # test unit_cooldown
  assert get_night_count_by_dist(0, 15, 1, cooldown) == 0
  assert get_night_count_by_dist(0, 15, 2, cooldown) == 1
  assert get_night_count_by_dist(0, 15, 3, cooldown) == 2
  assert get_night_count_by_dist(0, 15, 4, cooldown) == 5

  # test initial turn
  assert get_night_count_by_dist(1, 15, 0, cooldown) == 0
  assert get_night_count_by_dist(2, 15, 0, cooldown) == 1
  assert get_night_count_by_dist(3, 15, 0, cooldown) == 2
  assert get_night_count_by_dist(4, 15, 0, cooldown) == 5


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

  city = City(0, 'c_10', fuel=5, light_upkeep=2)
  assert not city_wont_last_at_nights(38, city)


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
