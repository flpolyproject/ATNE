

import os, sys, glob
import traci
import traci.constants as tc
from settings import Settings
from random import choice
sys.path.append("./../")
from traci_env import EnvironmentListener, BaseEnv
from visualize import Visualize


from uav import UAV


def start_simulation(Env, sim_number, gui=False, _seed=None):

	if gui:
		traci.start(["sumo-gui", "-c", Settings.sumo_config])
	else:
		traci.start(["sumo", "-c", Settings.sumo_config])


	
	env = Env(sim_number=sim_number, _seed=_seed)
	my_vis = Visualize(env)

	while True:

		traci.simulationStep()
		traci.addStepListener(env)
		if gui:
			my_vis.show()  #this is for visualization of the path
		if env.break_condition:
			break


	print("veh succesffuly arrived ", env.sim_env.success_veh)
	traci.close()


	env.post_process.to_csv()

	return env.sim_env


def run(gui=False, number=1, Env=EnvironmentListener):
	_seed=3

	sim_number = 0
	while sim_number < number:

		try:
			start_simulation(Env, sim_number, gui=gui, _seed=_seed)
			sim_number+=1
		except traci.exceptions.TraCIException as e:
			print("Restarting simulation failed at number ", sim_number, e)
			traci.close()
			continue
	start_simulation(BaseEnv, sim_number+1, gui=gui, _seed=_seed)



if __name__ == '__main__':
	print(traci.__file__)
	run(gui=True, number=1, Env=EnvironmentListener)
	'''
	try:
		run(gui=True, number=50, Env=EnvironmentListener)
	except Exception as e:
		print("PROGRAM FAILED EXIT CODE ", e)

	finally:
		traci.close()
	'''


	




'''

 traci.start(["sumo", "-c", "sim1.sumocfg"], label="sim1")
 traci.start(["sumo", "-c", "sim2.sumocfg"], label="sim2")
 conn1 = traci.getConnection("sim1")
 conn2 = traci.getConnection("sim2")
 conn1.simulationStep() # run 1 step for sim1
 conn2.simulationStep() # run 1 step for sim2
#parallel sims
'''
