
import argparse
import os
import subprocess
from multiprocessing import Pool


import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Run a batch of matches given a dataset.')
parser.add_argument('--dataset', help='path to dataset of (map_size, agent_id)')
parser.add_argument('--base', help='base agent main.py path')
parser.add_argument('--feature', help='feature agent main.py path')
parser.add_argument('--work_dir', help='output dir')
parser.add_argument('-C', '--concurrency', default=4, type=int, help='concurent match count')

args = parser.parse_args()


def run_match(map_seed, feature_agent_id, agent_base, agent_feature, work_dir, timeout=15000):
  if feature_agent_id == 0:
    agent_base, agent_feature = agent_feature, agent_base

  cmd = f"lux-ai-2021 --seed {map_seed} {agent_base} {agent_feature} --python python3 --maxtime {timeout} "
  subprocess.run(cmd, cwd=work_dir, shell=True, check=True)


def run(args):
  run_match(*args)


df = pd.read_csv(args.dataset)
match_args = [(i.map_seed, i.agent_id, args.base, args.feature, args.work_dir)
              for i in df.itertuples()]
if not os.path.exists(args.work_dir):
  os.makedirs(args.work_dir)
with Pool(args.concurrency) as pool:
  for _ in tqdm(pool.imap_unordered(run, match_args)):
    pass
