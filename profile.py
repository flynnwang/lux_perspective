import json
import os

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput
from kaggle_environments import make

from agent import agent

env = make("lux_ai_2021",
           configuration={
               "loglevel": 2,
               "annotations": True,
               'episodeSteps': 100,
               "actTimeout": 30,
               "seed": 222071549
           },
           debug=True)

config = Config(max_depth=15)
graphviz = GraphvizOutput(output_file="agent.perf_v6.png")
with PyCallGraph(output=graphviz, config=config):
  env.run(["simple_agent", agent])
