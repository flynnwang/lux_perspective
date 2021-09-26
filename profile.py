import json
import os

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput
from kaggle_environments import make

from agent import agent

env = make("lux_ai_2021", configuration={"loglevel": 2, "annotations": True,
                                         'episodeSteps': 75,
                                         "actTimeout": 30}, debug=True)

config = Config(max_depth=20)
graphviz = GraphvizOutput(output_file="agent.perf_v5.png")
with PyCallGraph(output=graphviz, config=config):
  # trainer = env.train(["simple_agent", agent])
  env.run(["simple_agent", agent])
