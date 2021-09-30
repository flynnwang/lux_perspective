import json
import os

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput
from kaggle_environments import make

from agent import agent

PLAYER_ID = 0
MAP_SEED = 222071549
# replay_path = '/Users/flynnwang/dev/playground/box_replays/run_2021_0928_24/replays/1632868916169_D0mUqJQ71BJb.json'
replay_path = "/Users/flynnwang/dev/playground/box_replays/replays/1632970563111_VID7udiafe0j_stateful.json"

env = make("lux_ai_2021",
           configuration={
               "loglevel": 2,
               "annotations": True,
               'episodeSteps': 50,
               "actTimeout": 30,
               "seed": MAP_SEED
           },
           debug=True)


class Replayer:

  def __init__(self, agent, replay_json, player_id=0):
    self.agent = agent
    self.replay_json = replay_json
    self.player_id = player_id
    self.env = env
    self.step = 0
    self.states = self.replay_json['stateful']

  def play(self, steps=1, print_step=False):
    for i in range(steps):
      if print_step:
        print("Step %s" % self.step)

      self.simulate(i)
      self.step += 1

  def simulate(self, step):
    step_state = self.states[step]
    obs = {"step": step, "updates": step_state}
    self.agent(obs, env.configuration)


with open(replay_path, 'r') as f:
  replay_json = json.loads(f.read())

__import__('pdb').set_trace()

r = Replayer(agent, replay_json, player_id=PLAYER_ID)
r.play(100)

config = Config(max_depth=16)
graphviz = GraphvizOutput(output_file="agent.perf_v5.png")
with PyCallGraph(output=graphviz, config=config):
  r.play(10)
