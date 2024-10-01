import math
from enum import Enum

import mesa
import networkx as nx


class State(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RESISTANT = 2


def number_state(model, state):
    return sum(1 for a in model.grid.get_all_cell_contents() if a.state is state)


def number_infected(model):
    return number_state(model, State.INFECTED)


class VirusOnNetwork(mesa.Model):
    """
    A virus model with some number of agents
    """

    def __init__(
            self,
            num_nodes=10,
            avg_node_degree=3,
            initial_outbreak_size=1,
            virus_spread_chance=0.4,
            virus_check_frequency=0.4,
            recovery_chance=0.3,
            gain_resistance_chance=0.5,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        prob = avg_node_degree / self.num_nodes
        self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=prob)
        self.grid = mesa.space.NetworkGrid(self.G)

        self.initial_outbreak_size = (
            initial_outbreak_size if initial_outbreak_size <= num_nodes else num_nodes
        )
        self.virus_spread_chance = virus_spread_chance
        self.virus_check_frequency = virus_check_frequency
        self.recovery_chance = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance

        # DataCollector now uses lambda functions for counting agent states.
        self.datacollector = mesa.DataCollector(
            {
                "Infected": lambda m: number_state(m, State.INFECTED),
                "Susceptible": lambda m: number_state(m, State.SUSCEPTIBLE),
                "Resistant": lambda m: number_state(m, State.RESISTANT),
            }
        )

        # Use `enumerate()` to provide unique IDs for each agent.
        for i, node in enumerate(self.G.nodes()):
            a = VirusAgent(
                unique_id=i,  # Ensure each agent gets a unique identifier
                model=self,
                initial_state=State.SUSCEPTIBLE,
                virus_spread_chance=self.virus_spread_chance,
                virus_check_frequency=self.virus_check_frequency,
                recovery_chance=self.recovery_chance,
                gain_resistance_chance=self.gain_resistance_chance,
            )

            # Add the agent to the node in the grid
            self.grid.place_agent(a, node)

        # Infect some nodes
        infected_nodes = self.random.sample(list(self.G), self.initial_outbreak_size)
        for a in self.grid.get_cell_list_contents(infected_nodes):
            a.state = State.INFECTED

        # Use RandomActivation to handle agent activation
        self.schedule = mesa.time.RandomActivation(self)

        # Add each agent to the schedule for activation management
        for agent in self.grid.get_all_cell_contents():
            self.schedule.add(agent)

        self.running = True
        self.datacollector.collect(self)

    # Simplified resistant to susceptible ratio calculation
    def resistant_susceptible_ratio(self):
        susceptible = number_state(self, State.SUSCEPTIBLE)
        return number_state(self, State.RESISTANT) / susceptible if susceptible > 0 else math.inf

    def step(self):
        # Step through all agents using the scheduler and collect data
        self.schedule.step()
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in range(n):
            self.step()


class VirusAgent(mesa.Agent):
    """
    Individual Agent definition and its properties/interaction methods
    """

    def __init__(
            self,
            unique_id,
            model,
            initial_state,
            virus_spread_chance,
            virus_check_frequency,
            recovery_chance,
            gain_resistance_chance,
    ):
        super().__init__(unique_id, model)  # Use unique_id as required by mesa

        self.state = initial_state

        self.virus_spread_chance = virus_spread_chance
        self.virus_check_frequency = virus_check_frequency
        self.recovery_chance = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance

    # Helper function to get susceptible neighbors to reduce repetitive code
    def get_susceptible_neighbors(self):
        neighbors_nodes = self.model.grid.get_neighborhood(self.pos, include_center=False)
        return [
            agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes)
            if agent.state is State.SUSCEPTIBLE
        ]

    # Infect susceptible neighbors based on virus spread chance
    def try_to_infect_neighbors(self):
        for a in self.get_susceptible_neighbors():
            if self.random.random() < self.virus_spread_chance:
                a.state = State.INFECTED

    def try_gain_resistance(self):
        if self.random.random() < self.gain_resistance_chance:
            self.state = State.RESISTANT

    def try_remove_infection(self):
        # Try to remove
        if self.random.random() < self.recovery_chance:
            # Success
            self.state = State.SUSCEPTIBLE
            self.try_gain_resistance()

    def try_check_situation(self):
        if (self.random.random() < self.virus_check_frequency) and (
            self.state is State.INFECTED
        ):
            self.try_remove_infection()

    def step(self):
        if self.state is State.INFECTED:
            self.try_to_infect_neighbors()
        self.try_check_situation()
