from settings import GraphSetting
import traci
#from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing
from traci_env import EnvironmentListener, BaseEnv
from settings import GraphSetting
import numpy as np
import random
import parallel
import pandas as pd
'''

def get_travel_time(tup):
	traci.start(["sumo", "-c", GraphSetting.sumo_config])
	travel_time = traci.simulation.findRoute(tup[0], tup[1])
	traci.close()
	return travel_time.travelTime



if __name__ == "__main__":
	traci_env_obj = EnvironmentListener(sim_number=0, _seed=None, init=False)
	edges = list(traci_env_obj.sim_env.map_data.edges.keys())
	random_junctions = [(random.choice(edges), random.choice(edges)) for x in range(100)]

	print(random_junctions, sep="\n")

	with multiprocessing.Pool(5) as pol:
		result = pol.map(get_travel_time, random_junctions)

		print(result)

'''
'''

traci_env_obj = EnvironmentListener(sim_number=0, _seed=None, init=False)
edges = list(traci_env_obj.sim_env.map_data.edges.keys())
random_junctions = [(random.choice(edges), random.choice(edges)) for x in range(100)]

myp = parallel.MyPool()

'''

from itertools import combinations

'''


dt_str = 'i, i,'

for i in range(2, 50):
	dt = np.dtype(dt_str)

	print(f"generating for {i} dstr {dt_str}")
	np.fromiter(combinations(range(50),i),dtype= dt, count=-1)
	dt_str += 'i,'


'''

#using binomial distribution, treating each veh as an indepent trial

#ncr 50c{1:50}

# how far is our algorithm is away from nash equilibrium

'''
import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

#50 veh
total = 0
for i in range(1, 51):
	total += ncr(50, i)

print(total)

'''


class myobj(object):
	def __init__(self, value):
		self.value = value
	def __repr__(self):
		return f"{self.value}"

'''
class test(object):
	def __init__(self):
		self.testdict = {"test1":[myobj(3), myobj(1), myobj(7)]}

		#for key, valuein in self.testdict.items():
		#	self.myfunc(valuein)

	def myfunc(self, mylist):
		testlist = mylist.copy()

		sorted_player_list = sorted(testlist, key=lambda x: x.value)
		for myv in sorted_player_list:
			myv.value = 8



result = test()
print(result.testdict["test1"])


test = 0

if test:
	print("wow")

'''

