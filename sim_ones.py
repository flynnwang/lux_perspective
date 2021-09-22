
def sim_one_turn_at_cell(worker, next_cell, game):
  """Sim one turn for worker move onto some cell, return cargo.

  1) collect resource
  2) drop resource to city.
  3) if night, make Units consume resources and CityTiles consume fuel
    - if unit no resource, removed
    - if city gone, unit gone with it
  """
  cargo = worker.cargo
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

  # count the number of neighbour resource types
  collect_cells = (get_neighbour_positions(next_cell.pos, game.game_map, return_cell=True)
                   + [next_cell])
  res_type_to_cells = defaultdict(list)
  for c in collect_cells:
    if c.resource:
      res_counter[c.resource.type.upper()].append(c)

  # For each type of resource, collect resource (asume no other wokers)
  for res_type in ALL_RESOURCE_TYPES:
    res_cells = res_counter[res_type]
    req_amt = request_amount(cargo, res_type, len(res_cells))
    if req_amt == 0:
      continue

    collect_amt = 0
    for c in res_cells:
      collect_amt += min(c.resource.amount, req_amt)

    add_resource_to_cargo(cargo, collect_amt, res_type)

  # 2) if worker on city tile, put its fuel into the city
  unit_on_citytile = next_cell.citytile != None
  if unit_on_citytile:
    city = game.player.cities[next_cell.citytile.cityid]
    city_last = city_last_days()
    city_upkeep = city.light_upkeep

    # TODO: check if city will gone.
    cargo = Cargo()
  




