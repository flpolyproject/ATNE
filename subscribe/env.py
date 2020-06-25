import numpy as np
from player import Player
import heapq
from _map import Map
import traci.constants as tc
from random import choice, randrange
from settings import GraphSetting
from multiprocessing import cpu_count, Manager, Queue, Pool
import traci
from operator import itemgetter
import itertools
from util import *
from functools import reduce

import time as tm
from concurrent.futures import ThreadPoolExecutor as pool
import threading

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import logging
#todo:
'''

handling when junction name is cluster cant find path: done
fix inside setbucket to properly getting all the edges even with : done
fix map parsing to handle : done
fix vehicle not arriving at final destination: the problem is that the vehicle is also setting next poi to current poi


'''

#improvements
'''
sensing plan often calculated with only 1 player at the poi: possible solution harish algo
not helping with road utilization because each vehcle takes similar path, maybe can figure out a way to smartly distribute the rewards to max util
the esp and eu values are too low

consider deviation cost, this will allow the vehicle to see that larger picture, maybe planning multiple poi ahead of time
as of right now the vehicle is only considering one poi at a time without considering its original path

reward adjustments regarding to time, solve the 1 vehicle problem, allowing vehicles to have a more diverse route
also adds more chaos on top of only considering the number of possible player being there





for temperal coverage create a function interms of average time gap collected at poij

'''






class Environment(object):
	def __init__(self):

		
		self.map_data = Map(GraphSetting.sumo_config)
		self.player_list = {}
		self.poi_to_junct = {}  #enter poi id to get junction, enter junction to get poi id
		self.index_counter = 0

		

		self.poi_list = {}     #poi {poi_key:{veh_key:veh_value}} to keep track of if vehcle is being tracked by the poi to know if
		#veh is leaving or entering junction


		self.success_veh = [] #vehicle success arrived dest
		self.poi_que = {} # {poiid:{vehid:edge, vehid:edge}} #when player arived to raidus add to this, every 10 sec this is cleared


		#below are the variables for guis
		self.veh_data = None #result of veh storing location
		self.track_veh = None #id storing the vehicle to track


	def stop_vehicle_handle(self, t): #handle when vehicle is stopped calculate the sensing plan with buffered vehicles
		if (t % GraphSetting.buffer_interval) == 0: #the number of sec passed by
			#print("im checking for que ", self.t)
			if self.poi_que:
				for poi_key, veh_waiting_list in self.poi_que.items():
					sp, number_players = self.update_veh_collection_status(veh_waiting_list, poi_key) #this is only called when veh is being handled at poi
					reward_value = (self.map_data.pois[poi_key].value/pow(number_players, 2))

					for veh, edge in veh_waiting_list.items():

						self.adjust_sensing_plan(poi_key, veh, sp, edge, reward_value) #when veh route is adjusted does this really need to be adjusted for every veh or maybe it should be only per poi

						#self.player_list[veh].reward += (self.map_data.pois[poi_key].value/pow(len(veh_waiting_list), 2))

						

						try:
							traci.vehicle.setStop(veh, edge, duration=0)
							#print(f"i succeeded at resuming {veh}. routes: {traci.vehicle.getRoute(veh)}, index : {traci.vehicle.getRouteIndex(veh)}, current: {traci.vehicle.getRoute(veh)[traci.vehicle.getRouteIndex(veh)]} shouldbe {edge}")
						except traci.exceptions.TraCIException as e:
							print(f"i failed at resuming {veh}. routes: {traci.vehicle.getRoute(veh)}, index : {traci.vehicle.getRouteIndex(veh)}, current: {traci.vehicle.getRoute(veh)[traci.vehicle.getRouteIndex(veh)]} shouldbe {edge}")
							#traci.vehicle.setStop(veh, edge, duration=0)
							self.track_veh = veh

							#traci.vehicle.setStop(veh, edge, duration=0)
							raise traci.exceptions.TraCIException("i failed here wtf")
						

						#traci.vehicle.setStop(veh, traci.vehicle.getRoute(veh)[traci.vehicle.getRouteIndex(veh)], duration=0)
						print(f"moving {veh} at sim step {t}, processed sp for {len(veh_waiting_list)} player(s)")
				self.poi_que = {}






	def calculate_next_poi_new(self, veh_id, current_node, add=False):#add to show that its initializing
		#this is called when player is added and is called every time a play arrived at a poi
		#loops through every player and every poi to find the player prob for every poi


		#print("current node is ", current_node)



		player = self.player_list[veh_id]


		max_eu = 0
		max_eu_location=None
		final_distance_cap = None

		player_data = self.player_list if add else traci.vehicle.getAllSubscriptionResults()
		len_player = len(player_data)
		

		for poi_id, self.map_data.pois[poi_id] in self.map_data.pois.items():

			assert len_player >0, 'something is wrong no player in system'



			if not poi_id in self.player_list[veh_id].poi_potential:
				#player cant go to this poi due to distance capacity
				continue
			else:
				print(f"{veh_id} considering going to {poi_id}")


			total_cost = self.map_data.pois[poi_id].player_potential[veh_id] + self.player_list[veh_id].poi_potential[poi_id]



			assert total_cost <= self.player_list[veh_id].distance_capacity, f"failed {veh_id} dist cap {self.player_list[veh_id].distance_capacity} < {total_cost} going to {poi_id}"
			assert poi_id in self.player_list[veh_id].current_poi_distribution, f"failed {poi_id} not in {veh_id} distribution"





			if len_player <=1:  #im the only one there
				e_u = self.map_data.pois[poi_id].value
				e_sp = 1
			else:

				e_u, e_sp = self.calculate_utility(veh_id, poi_id, self.map_data.pois[poi_id].value, player_data)


			if (e_u > max_eu) and (e_sp <= player.capacity) and (self.map_data.pois[poi_id].junction!=player.target_poi):
				max_eu = e_u
				max_eu_location = self.map_data.pois[poi_id].junction
				final_distance_cap = total_cost


			#reset poi potential players or it might affect next time bucket generation
			#self.map_data.pois[poi_id].player_potential = {}
				


		if final_distance_cap:
			self.player_list[veh_id].distance_capacity -= final_distance_cap

		#self.player_list[veh_id].poi_potential = {}

		return max_eu_location



	def calculate_utility_new(self, veh_id, poi_id, reward, player_data):
		'''
		calculate the expected util for particular i veh_id to poi_id
		sum up all the times
		'''


		#iteration through poi_combintaions and populate
		#self.set_combs(poi_id)


		
		player_data_keys = list(player_data.keys())


		time_eu_dict = {} #time as key m_eu_dict as value
		time_esp_dict = {}

		for index, time in enumerate(self.player_list[veh_id].current_poi_distribution[poi_id][1]):

			#filter out the ones that are not going there that means i have to regenerate combinations...... thats even more costly
			#combinations are assumed that all players have some probalit
			#even if i remove the player from the consideration list

			#self.filter_participants()

			for m, combs in self.player_list[veh_id].combinations.items(): #m is r {1:[10c1], 2:[10c2]} r 1:n
				prob = 1
				m_eu_dict = {} #number of players as key, eu as value
				m_esp_dict = {}
				for key, value in player_data.items(): #calculate the prob first then filter out the one that are not likely
					if key in combs:
						arriving_prob = self.find_probability(time, self.player_list[key].current_poi_distribution[poi_id])
						if key == veh_id:
							assert self.player_list[veh_id].current_poi_distribution[poi_id][0][index] == arriving_prob,\
								f"time dist doesnt match {veh_id} {self.player_list[veh_id].current_poi_distribution[poi_id][0][index]} {arriving_prob}"
					else:
						arriving_prob = (1 - self.find_probability(time, self.player_list[key].current_poi_distribution[poi_id]))


					prob *= arriving_prob
				
				eu = prob*(reward/pow(m, 2))
				esp = prob*self.compute_sensing_plan(m, (reward/pow(m, 2)), self.map_data.junctions[self.map_data.pois[poi_id].junction].cost)
				m_eu_dict[m] = eu
				m_esp_dict[m] = esp

			time_eu_dict[time] = m_eu_dict
			time_esp_dict[time] = m_esp_dict


		
		total_eu = sum([sum(_m.values()) for _time, _m in time_eu_dict.items()])

		total_esp = sum([sum(_m.values()) for _time, _m in time_esp_dict.items()])

		

		print(f"for veh {veh_id} to poi {poi_id} eu is {total_eu} esp is {total_esp} current cap:{self.player_list[veh_id].capacity}")
		return total_eu, total_esp





	def calculate_utility(self, veh_id, poi_id, reward, player_data):
		'''
		calculate the expected util for particular i veh_id to poi_id
		sum up all the times
		'''


		#iteration through poi_combintaions and populate

		self.player_list[veh_id].combinations = defaultdict(list)  #reset player combinations

		self.set_combs(poi_id) #setting the combinations for all the players that are potentially going to this poi
		player_data = self.map_data.pois[poi_id].player_potential

		player_data_keys = list(player_data.keys())


		time_eu_dict = {} #time as key m_eu_dict as value
		time_esp_dict = {}

		for index, time in enumerate(self.player_list[veh_id].current_poi_distribution[poi_id][1]):

			#filter out the ones that are not going there that means i have to regenerate combinations...... thats even more costly
			#combinations are assumed that all players have some probalit
			#even if i remove the player from the consideration list

			#self.filter_participants()

			for m, combs in self.player_list[veh_id].combinations.items(): #m is r {1:[10c1], 2:[10c2]} r 1:n #this is set using set_combos
				prob = 1
				m_eu_dict = {} #number of players as key, eu as value
				m_esp_dict = {}
				for key, value in player_data.items(): #iteration of all players potentially going to poi
					if key in combs:
						arriving_prob = self.find_probability(time, self.player_list[key].current_poi_distribution[poi_id])
						if key == veh_id:
							assert self.player_list[veh_id].current_poi_distribution[poi_id][0][index] == arriving_prob,\
								f"time dist doesnt match {veh_id} {self.player_list[veh_id].current_poi_distribution[poi_id][0][index]} {arriving_prob}"
					else:
						arriving_prob = (1 - self.find_probability(time, self.player_list[key].current_poi_distribution[poi_id]))


					prob *= arriving_prob
				
				eu = prob*(reward/pow(m, 2))
				esp = prob*self.compute_sensing_plan(m, (reward/pow(m, 2)), self.map_data.junctions[self.map_data.pois[poi_id].junction].cost)
				m_eu_dict[m] = eu
				m_esp_dict[m] = esp

			time_eu_dict[time] = m_eu_dict
			time_esp_dict[time] = m_esp_dict


		
		total_eu = sum([sum(_m.values()) for _time, _m in time_eu_dict.items()])

		total_esp = sum([sum(_m.values()) for _time, _m in time_esp_dict.items()])

		

		print(f"for veh {veh_id} to poi {poi_id} eu is {total_eu} esp is {total_esp} current cap:{self.player_list[veh_id].capacity}")
		return total_eu, total_esp





	def generate_bucket(self, veh_id=None):

		#when veh arrive at destination the bucket should be changed for each veh the same combination no longer apply


		#this also sets the potential players to every potential pois based on distance cap

		#this is caled everytime player arrive at a poi

		def set_bucket(veh_id, current_edge):

			print(f"Generating buckets for {veh_id}....")

			#for each veh need to reset their dic before generating buckets

			

			for poi_id, poi_value in self.map_data.pois.items():
				key = self.map_data.pois[poi_id].junction
				value = self.map_data.pois[poi_id].value

				
				route_value = self.map_data.find_route_reroute(current_edge, key)
				
				route_value_todest = self.map_data.find_best_route(key, self.player_list[veh_id].dest_junc)

				total_time = route_value.travelTime + route_value_todest.travelTime

				if self.player_list[veh_id].distance_capacity < total_time:
					continue
					#the player combination to poi is not updated

				print(f"considering {poi_id} total cost {total_time} cap left {self.player_list[veh_id].distance_capacity}")

				self.player_list[veh_id].poi_potential[poi_id] = route_value.travelTime
				self.map_data.pois[poi_id].player_potential[veh_id] = route_value_todest.travelTime
				


				route_edges = route_value.edges
				self.player_list[veh_id].temp_edges[key]=route_edges



				new_mean = sum([self.map_data.edges[e].distance/self.map_data.edges[e].speed for e in route_edges if not ':' in e]) #sum of the means of edges within the route to each poi
				new_std = reduce(lambda x,y:np.sqrt(x**2+y**2), [self.map_data.edges[e].std for e in route_edges if not ':' in e]) # combine the std of all the edges

				#: is for junctions, when vehicle in motion, tc.roadid can return junction

				route_array = generate_speed_distribution(new_mean, new_std) #distribution data generated based on new mean and std

				result = generate_speed_bucket(route_array, bins_num=6) #generate histogram with bin number

				self.player_list[veh_id].current_poi_distribution[poi_id] = result #save this histogram information to the player object

				#current poi_distribution {poi_id:histogram}
	



		if veh_id: #finding dict for only 1 vehicle
			set_bucket(veh_id, self.player_list[veh_id].current_edge)
			#self.set_combinations_new(add=True)
			
		else: #for when 1 vehicle arrived at a poi need to evaluate the next poi thus need to update every other players bucket



			for poi_id, poi_value in self.map_data.pois.items():
				poi_value.player_potential = {}

			for veh_id, veh_value in traci.vehicle.getAllSubscriptionResults().items():
				self.player_list[veh_id].poi_potential = {}

			for veh_id, veh_value in traci.vehicle.getAllSubscriptionResults().items():
				set_bucket(veh_id, veh_value[tc.VAR_ROAD_ID])

			#self.set_combinations_new()

			#because we are in the player loop after updating the potential pois for this particular player, go ahead and generate the combination for this player

	
		
	def set_combs(self, poi_id, add=False): #setting the combinations of those players who are potentially able to participate in this poi
		total_players = len(self.map_data.pois[poi_id].player_potential)
		print(f"{poi_id} is generating combinations for {total_players}")
		for i in range(total_players):
			combs = itertools.combinations(list(self.map_data.pois[poi_id].player_potential.keys()), i+1) #try optimizing using numpy or iterators instread

			#combs = 
			self.map_data.pois[poi_id].combinations[i+1] = combs
			self.set_combinations_player(i+1, combs) #setting combs for theplayers based off the comb


	






	def set_bucket_new(self, veh_id, current_edge, add=True):
		#print(f"setting histogram for {veh_id} add: {add}")
		for poi_id, self.map_data.pois[poi_id] in self.map_data.pois.items():
			key = self.map_data.pois[poi_id].junction
			value = self.map_data.pois[poi_id].value

			route_edges = self.map_data.find_route_reroute(current_edge, key).edges
			self.player_list[veh_id].temp_edges[key]=route_edges


			new_mean = sum([self.map_data.edges[e].distance/self.map_data.edges[e].speed for e in route_edges if not ':' in e]) #sum of the means of edges within the route to each poi
			new_std = reduce(lambda x,y:np.sqrt(x**2+y**2), [self.map_data.edges[e].std for e in route_edges if not ':' in e]) # combine the std of all the edges

			#: is for junctions, when vehicle in motion, tc.roadid can return junction

			route_array = generate_speed_distribution(new_mean, new_std) #distribution data generated based on new mean and std

			result = generate_speed_bucket(route_array, bins_num=6) #generate histogram with bin number

			self.player_list[veh_id].current_poi_distribution[poi_id] = result #save this histogram information to the player object

			
			


	def compute_sensing_plan(self, player_amount, reward, cost):
		#print('player amount is ', player_amount)
		sensing_plan = ((player_amount-1)*reward)/((player_amount**2)*cost)

		#print('sensning plan value is ', sensing_plan)
		return sensing_plan


	def print_pc(self):
		for key, value in self.player_list.items():
			print(value.combinations)

			#value.combinations = defaultdict(list)


	def set_combinations_player(self, i, combs):
		
		'''



		np_combs = np.array(combs) #each combination is a row
		print(np_combs)
		for player_id, player_value in self.player_list.items():
			#print(f"procssing {player_id} is at {np.apply_along_axis(lambda x: player_id in x, 1, np_combs)}")
			
			player_value.combinations[i].append(np_combs[(np.apply_along_axis(lambda x: player_id in x, 1, np_combs)), :])

		print("first combintation list")
		self.print_pc()
		#exit()

		'''


		while True:
			try:
				comb = next(combs)#combs.pop()
				#print(f"combinations for {i} players {comb}")
				for player in comb:
					self.player_list[player].combinations[i].append(comb)
			except StopIteration as e:
				break



	def set_combinations(self, add = False):
		#this gets combinations for all the players after all the players has been initialized
		#intialize combination
		#this need to be fixed for memory error cant store all combinations for every number of vehicles
		#where should the poi potential players be populated, shoudl be inside calculate next poi new, but if its populated there then combinations should be generated per poi based


		'''
		
		print(f"All players added len: {len(self.player_list)} generating combinations...")
		player_keys = list(self.player_list.keys())
		all_combs = {}


		for i in range(len(self.player_list)):
			#print("generating combinations")
			combs = list(itertools.combinations(player_keys, i+1))
			#all_combs[i+1] = combs
			#print("setting combinations")
			self.set_combinations_player(i+1, combs)

		'''

		if add:

			for player_id, player_value in self.player_list.items():

				self.next_poi_reroute(player_id, player_value.start, player_value.prev_junction, add=add)





	def find_probability(self, time, distribution):
		'''
		given time and distribution of a veh at poi, find the prob of that time for the veh
		[0] is the probability, [1] is the intervals/bins
		'''
		buckets = distribution[1]

		try:
			upper_index = np.min(np.where(buckets>=time)) #index of upper boundary of time
			lower_index = np.max(np.where(buckets<=time)) #index of lower boundary of time	

			if upper_index == lower_index and upper_index==len(distribution[0]):
				lower_index -= 1



			#print(f'time searching for is {time}, upper:{upper_index}, lower:{lower_index}')
			#print(f'bucket is:', buckets)	
			#print(f'prob is:', distribution[0])	
		except ValueError:
			lower_index = None

		if not lower_index:
			return 0
		else:
			return distribution[0][lower_index]




			


	def add_player(self, veh_id, routes, dest_junc):  #this is called before setting combinations
		assert not veh_id in self.player_list, f"failed more than one player with {veh_id}"
		assert self.index_counter == int(veh_id.split('_')[1]), 'player id doesnt match counter'

		route_edges = routes.edges

		self.player_list[veh_id] = Player(veh_id, route_edges, self.map_data.edges[route_edges[0]]._from, dest_junc)
		self.player_list[veh_id].capacity = get_truncated_normal(GraphSetting.player_capacity_random[0], GraphSetting.player_capacity_random[1], 0, GraphSetting.player_capacity_random[0]*2).rvs(1)[0]


		try:
			print(f"{veh_id} shortest path travel time {routes.travelTime}")




			if GraphSetting.distance_capacity[0] == GraphSetting.distance_capacity[1]:
				self.player_list[veh_id].distance_capacity = (GraphSetting.distance_capacity[0] * routes.travelTime)
			else:
				self.player_list[veh_id].distance_capacity = np.random.randint(routes.travelTime * GraphSetting.distance_capacity[0], routes.travelTime * GraphSetting.distance_capacity[1])
		except ValueError:
			self.player_list[veh_id].distance_capacity = 0

		self.generate_bucket(veh_id=veh_id)


		self.index_counter+=1

		print(f"Added player {veh_id}, dist_cap: {self.player_list[veh_id].distance_capacity}")



	def reroute(self, veh_id, current_edge, upcome_edge, destination, add=False):
		try:
			print(f'{veh_id} traveling on {upcome_edge} change direction going towards {destination}({self.poi_to_junct[destination]})')
		except KeyError:
			print(f'{veh_id} traveling on {upcome_edge} change direction going towards {destination}(Destination)')
		print()

		shortest_route = self.map_data.find_route_reroute(upcome_edge, destination)
		

		shortest_route = list(shortest_route.edges)

		traci.vehicle.changeTarget(veh_id, shortest_route[-1])



		return shortest_route



		

	def update_capacity(self, veh, esp):

		try:
			#assert self.player_list[veh].capacity >= esp, f"CAPACITY change ERROR cap:{self.player_list[veh].capacity} esp:{esp}"
			if esp > self.player_list[veh].capacity and esp == 1: #in the case of 1 player esp is set to 1 but still higher than 
				self.player_list[veh].capacity = 0
			else:
				self.player_list[veh].capacity -= esp
		except KeyError:
			print(veh_value, 'Error')






	def update_veh_collection_status(self, veh_value, poi_key):
		#iterate through all the vehicles
		'''
		this function have issue when the diff of esp(i) and esp(i-1)
		eg. if esp(2) is 5.3 esp(1) is 1, player capacitys are 3 and 20
		when calc esp of 2, 1 fit. but when calc esp of 1, 2 fits

		the len(veh_value) is mostly 1, only when multiple vehicles arrives in the same radius will the length >1

		'''
		keys_list = list(veh_value.keys())
		cap_list = []
		veh_cap_list = []
		counter_list = []
		i_list = []

		temp_veh_value = veh_value.copy()

		for i in range(len(veh_value), 0, -1):
			esp = self.compute_sensing_plan(i, self.map_data.pois[poi_key].value, self.map_data.junctions[self.poi_to_junct[poi_key]].cost)
			cap_list.append(esp)
			veh_cap_list.append(self.player_list[keys_list[i-1]].capacity)
			counter = 0 #this to count how many fits the capacity
			min_cap_veh_id = None #only remove one veh with smallest capacity if not fit
			for new_key, new_value in veh_value.items():

				if esp<=self.player_list[new_key].capacity:
					counter+=1
					self.player_list[new_key].participation = True
				else:
					self.player_list[new_key].participation = False
					if not min_cap_veh_id:
						min_cap_veh_id = new_key
					else:
						if self.player_list[new_key].capacity < self.player_list[min_cap_veh_id].capacity:
							min_cap_veh_id = new_key


			if min_cap_veh_id:
				del temp_veh_value[min_cap_veh_id]
				veh_value = temp_veh_value

			if counter == i: #this line
				if i==1:
					esp = 1
				return esp, i




			counter_list.append(counter)
			i_list.append(i)

		print("I should not be here ")
		print(f'length is {len(veh_value)}, esp list:{cap_list}, cap list:{veh_cap_list} counter {counter_list}, ilist {i_list}')
		exit()

	def process_destination(self):

		arrived_id = traci.simulation.getArrivedIDList()
		if arrived_id:
			for veh_id in arrived_id:
				if not veh_id in self.success_veh:
					self.success_veh.append(veh_id)
					print(f"vehicle {veh_id} arrived at destination")


	def wait_in_radius(self, poi_key,veh_id):
		#print("before stop routes ", traci.vehicle.getRoute(veh_id))

		#this function is for stopping vehicles
		routes = traci.vehicle.getRoute(veh_id)
		route_index = traci.vehicle.getRouteIndex(veh_id)
		start_edge = routes[route_index]
		start_index = route_index
		while True:
			try:
				traci.vehicle.setStop(veh_id, routes[route_index])
			

				break
			except traci.exceptions.TraCIException:

				route_index += 1
			except IndexError:
				#because the stopping edge is determined before rerouting to he next poi, the route index might be out of range because its reached the poi and dk where to go
				#print(f"oh well im out of index trying to stop {veh_id} at {poi_key} starting {start_edge} index {start_index}")
				#print(f"Routes: {routes}")
				#exit()

				routes = self.map_data.find_route_reroute(start_edge, GraphSetting.destination).edges
				route_index = 0
				traci.vehicle.setRoute(veh_id,routes)

		print(f"stopping.... {veh_id} at {poi_key}({routes[route_index]})")


		
		edge = routes[route_index]
			
		try:

			if not self.poi_que[poi_key]:
				self.poi_que[poi_key]= {veh_id:edge}
			else:
				self.poi_que[poi_key][veh_id] = edge
		except KeyError as e:
			self.poi_que[poi_key]= {veh_id:edge}


		self.track_veh = veh_id



		#print(self.poi_que[poi_key])
		#print("after stop routes ", traci.vehicle.getRoute(veh_id))


	def adjust_sensing_plan(self, key, veh, sp, current_edge, reward_value):
		#key is the current poi key
		#self.player_list[veh].target_poi = self.map_data.pois[key].junction #incase veh accidentally encounter poi, need to update
		#if vehicle predetermined to go to destination but encounter a poi then update

		self.generate_bucket()

		if self.player_list[veh].participation:

			before_capacity = self.player_list[veh].capacity
			self.player_list[veh].reward += reward_value
			self.update_capacity(veh, sp)
			print(f"{veh} CAP_before:{before_capacity} CAP_after:{self.player_list[veh].capacity}: SP:{sp} at junction {key}({self.poi_to_junct[key]})")
			self.player_list[veh].participation = False

		else:

			print(f"{veh} not participating at {key} sp:{sp} cap:{self.player_list[veh].capacity}")


		self.next_poi_reroute(veh, current_edge, self.map_data.pois[key].junction)


		

	def next_poi_reroute(self, veh, current_edge, prev_junction, add=False): #this is called everytime we want to determine poi and reroute

		next_poi = self.calculate_next_poi_new(veh, prev_junction, add=add) #maybe this function can return none for going towards dest

		if not next_poi:
			#this is the weighted random jumps
			#right now set to go home if none fit
			print(f'{veh} reached max capacity, going towards destination')
			next_poi = self.player_list[veh].dest_junc
		else:

			self.player_list[veh].target_poi = next_poi

		st = self.reroute(veh, None, current_edge, next_poi)




	def process_poi(self):
		#should check is the vehicle is currently on the poi adjacent edges

		poi_data = traci.poi.getAllContextSubscriptionResults() #poi data is {poikey:{vehid:sub_info_veh}}
		if poi_data:
			for key, value in poi_data.items(): #loop through all poi list

				#generate bucket should be only for one player that arrived at junction

				#self.generate_bucket() #update players prob dist buckets for every reward #this is wrong should only update for that vehicle that arrived
				#sp, number_players = self.update_veh_collection_status(value, key) #update all the vehicles at the junction for if they are participating or not
				
				#print(f'esp is {esp}, {number_players}')
				for veh, veh_value in value.items(): #loop through all vehicles in junctions
					if not veh in self.poi_list[key] and self.player_list[veh].prev_poi != key: #this if statement for when vehicle first enter junction
						#update capacity and reward first
						#print('vehicle approaching poi', veh, key)

						#maybe i shouldnt check for if veh at target poi if accidentally stomble then still try to collect
						#current_edge = veh_value[tc.VAR_ROAD_ID]
						self.track_veh = veh

						self.wait_in_radius(key, veh) #this process poi function need to stop vehicle and everything else happens in the stop handle function for updating sp
					

						self.poi_list[key][veh] = veh_value #add vahicle in poi list for removal later

					elif veh in self.poi_list[key] and self.player_list[veh].prev_poi != key:
						#check if it should delete vehicle
						try:
							if self.map_data.edges[veh_value[tc.VAR_ROAD_ID]]._to != key:
								#print('vehicle left junction')
								self.player_list[veh].prev_poi = key
								del self.poi_list[key][veh]
								self.track_veh = None
						except KeyError:
							#print('reached junction')
							continue




if __name__ == '__main__':
	pass
	#problem, high expected sensing plan, but when 