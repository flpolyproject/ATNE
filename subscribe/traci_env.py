
import traci
import traci.constants as tc
from postprocess import PostGraph 
from env import Environment
from random import choice, seed
import math
from settings import GraphSetting
import dill as pickle
import os, glob
from util import *
from _map import Poi
import time as tm

from visualize import Visualize

#env is for storing data contains map and players
#try to keep all traci calls in here


class EnvironmentListener(traci.StepListener):
	def __init__(self, sim_number, init=True, _seed=None, ATNE=True):
		super(EnvironmentListener, self).__init__()
		seed(_seed)


		self.ATNE = ATNE

		self.sim_number = sim_number

		self.post_process = PostGraph(self.sim_number, columns=["sim_number", "sim_step", "veh_id", "edge_id", "speed", "capacity", "budget", "prev_poi"])
		self.t = 0

		self.break_condition = False #condition to check if all vehicles has arrived

		
		file_dir = os.path.dirname(GraphSetting.sumo_config)
		map_list = glob.glob(os.path.join(file_dir, r"*.map"))
		try:
			self.sim_env = self.read(map_list[0])
			print(f"loaded map from {map_list[0]}")
		except IndexError:
			print(f".map file generating for {GraphSetting.sumo_config}")
			self.sim_env = Environment()
			self.save(GraphSetting.sumo_config, self.sim_env)
			
		if init:

			self.initial_reward_random(GraphSetting.reward_numbers)
			self.initial_route_random(GraphSetting.car_numbers)
			self.junction_sub()


		
	def read(self, target_path):
		with open(target_path, "rb") as f:
			return pickle.load(f)
	def save(self, source_path, target_object):
		file_name = f"{os.path.basename(source_path).split('.')[0]}.map"
		file_dir = os.path.dirname(source_path)
		target_path = os.path.join(file_dir, file_name)
		with open(target_path, "wb") as f:
			pickle.dump(target_object, f)
			print(f"map saved to {target_path}")
		

	def initial_route_random(self, amount, seed=None):


		list_edges = list(self.sim_env.map_data.edges)
		list_juncts = list(self.sim_env.map_data.junctions)


		for i in range(amount):
			veh_id = 'veh_'+str(i)
			route_id = 'route_'+str(i)
			#traci.route.add(route_id, [choice(list_edges) for _ in range(2)])	
			#traci.route.add(route_id, [choice(list_edges), '-cell0_0N'])
			
			while True:
				try:
					start = choice(list_juncts)
					end = GraphSetting.destination
					if GraphSetting.destination == "random":
						end = choice(list_juncts)

					if start == end:
						continue

					route = self.sim_env.map_data.find_best_route(start, end)

					if not route.edges:
						continue


					break
				except traci.exceptions.TraCIException:
					continue

			route_edges = route.edges

			try:

				traci.route.add(route_id, route_edges)
				traci.vehicle.add(veh_id, route_id,departLane='random')

			except traci.exceptions.TraCIException:
				assert True, f"FAILED TO ADD ROUTE {veh_id}, edges:{route_edges}"


			

			self.sim_env.add_player(veh_id, route, end)

		#after all vehicles added
		if self.ATNE:
			self.sim_env.set_combinations(add=True)
			#self.sim_env.set_combinations_new(add=True)
			#combination is called after all players are added


	def initial_reward_random(self, amount): #initialize all the rewards

		#traci.junction.subscribeContext(GraphSetting.destination, tc.CMD_GET_VEHICLE_VARIABLE, 20, [tc.VAR_EDGES, tc.VAR_ROAD_ID])

		all_junctions = list(self.sim_env.map_data.junctions.keys())

		for i in range(amount):

			id_value = f'poi_{str(i)}'
			junction=choice(all_junctions)

			self.sim_env.map_data.pois[id_value] = Poi(junction, \
				get_truncated_normal(GraphSetting.player_reward_random[0], GraphSetting.player_reward_random[1], 0, GraphSetting.player_reward_random[0]*2).rvs(1)[0]) #add poi to dict with poi id as key

			#self.sim_env.poi_que[id_value] = {}
			self.sim_env.poi_list[id_value]= {}
			self.sim_env.poi_to_junct[id_value] = junction
			self.sim_env.poi_to_junct[junction] = id_value

			traci.poi.add(id_value, *traci.junction.getPosition(junction), color=(255,0,255,255), layer=10, height=10)
			#print(tuple(*traci.junction.getPosition(junction)))

			#Visualize.polygon(traci.junction.getPosition(junction), (255,0,255,255), 30)


			traci.poi.subscribeContext(id_value, tc.CMD_GET_VEHICLE_VARIABLE, GraphSetting.poi_radius, [tc.VAR_EDGES, tc.VAR_ROAD_ID])
			print(f'added {id_value} to location {junction}')

	def step(self, t=0):
		#action performed after each step aka simulation step

		self.vehicle_sub() #constant sub to veh to handle veh being added during simulation
		self.break_condition = self.populate_post() #for populating post processing data, and set break condition to get when all vehicle arrives
		self.sim_env.process_poi()
		self.sim_env.process_destination() #subscribe to destination of veh to make sure it arrives
		self.sim_env.stop_vehicle_handle(self.t)
		self.t+=1



		
	def populate_post(self): #for post processing and make sure vehicle all arrives
		self.sim_env.veh_data = traci.vehicle.getAllSubscriptionResults()
		if self.sim_env.veh_data:
			for veh_id, values in self.sim_env.veh_data.items():

				post_list = [self.sim_number, self.t, veh_id, values[tc.VAR_ROAD_ID], values[tc.VAR_SPEED], \
				self.sim_env.player_list[veh_id].capacity, self.sim_env.player_list[veh_id].reward, self.sim_env.player_list[veh_id].prev_poi\
				]

				self.post_process.append_row(post_list)

			return False
		
		return True








	def vehicle_sub(self):
		for veh_id in traci.vehicle.getIDList():
			traci.vehicle.subscribe(veh_id, [tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_EDGES, tc.VAR_ROUTE_INDEX,tc.VAR_ROAD_ID])


	def junction_sub(self):
		len_junction = 0
		for junc, junc_obj in self.sim_env.map_data.junctions.items():
			if not ':' in junc:
				dist_from_junc = EnvironmentListener.mean([self.sim_env.map_data.edges[x].distance for x in junc_obj.adjacent_edges_from])
				if dist_from_junc:
					#traci subscribe need to convert miles to meters
					traci.junction.subscribeContext(junc, tc.CMD_GET_VEHICLE_VARIABLE, (dist_from_junc/4)*1609.34, [tc.VAR_EDGES, tc.VAR_ROAD_ID])
					len_junction+=1
		#print('len junctions sub to ', len_junction) #show number of junc sub to



	@staticmethod
	def mean(list_value):
		if len(list_value) == 0:
			return
		return sum(list_value)/len(list_value)



class BaseEnv(EnvironmentListener):
	def __init__(self, sim_number, init=True, _seed=None, ATNE=False):
		super(BaseEnv, self).__init__(sim_number, init=init, _seed =_seed, ATNE=ATNE)

	def step(self, t=0):
		#action performed after each step aka simulation step

		self.vehicle_sub() #constant sub to veh to handle veh being added during simulation
		self.break_condition = self.populate_post() #for populating post processing data, and set break condition to get when all vehicle arrives
		#self.sim_env.process_poi()
		#self.sim_env.process_destination() #subscribe to destination of veh to make sure it arrives
		#self.sim_env.stop_vehicle_handle(self.t)
		self.t+=1


