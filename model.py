import mesa
import numpy as np
import random

from agent import *

class StreetView(mesa.Model):
    """
    Clase del modelo Flockers. Maneja la creación de agentes, su ubicación y programación.
    """

    def __init__(
        self,
        num,  # Cambia el nombre de population a num
        width=24,
        height=24,
        speed=1,
        vision=10,
        separation=2,
        cohere=0.025,
        separate=0.25,
        match=0.04,
        pathIndex = 0,
        buildingPos=[(2, 21), (3, 21), (4, 21), (5, 21), (6, 21), (7, 21), (8, 21), (9, 21),           (11, 21), (12, 21),                   (17, 21), (18, 21),                (21, 21), (22, 21),
                                     (3, 20), (4, 20), (5, 20), (6, 20), (7, 20), (8, 20), (9, 20), (10, 20), (11, 20), (12, 20),                   (17, 20),                          (21, 20), (22, 20),
                            (2, 19), (3, 19), (4, 19), (5, 19), (6, 19), (7, 19), (8, 19), (9, 19), (10, 19), (11, 19),                             (17, 19), (18, 19),                          (22, 19),
                            (2, 18), (3, 18), (4, 18), (5, 18), (6, 18),          (8, 18), (9, 18), (10, 18), (11, 18), (12, 18),                   (17, 18), (18, 18),                (21, 18), (22, 18),
                            
                            (2, 15), (3, 15), (4, 15),                   (7, 15),          (9, 15), (10, 15), (11, 15), (12, 15),                   (17, 15), (18, 15),                (21, 15), (22, 15),
                            (2, 14), (3, 14), (4, 14),                   (7, 14), (8, 14), (9, 14), (10, 14), (11, 14), (12, 14),                   (17, 14), (18, 14),                (21, 14),
                            (2, 13), (3, 13),                            (7, 13), (8, 13), (9, 13), (10, 13), (11, 13),                                       (18, 13),                (21, 13), (22, 13),
                            (2, 12), (3, 12), (4, 12),                   (7, 12), (8, 12), (9, 12), (10, 12), (11, 12), (12, 12),                   (17, 12), (18, 12),                (21, 12), (22, 12),
                            
                            
                            
                            
                            (2, 7), (3, 7), (4, 7), (5, 7),                           (8, 7), (9, 7), (10, 7), (11, 7), (12, 7),                    (17, 7), (18, 7), (19, 7), (20, 7), (21, 7), (22, 7),
                                    (3, 6), (4, 6), (5, 6),                           (8, 6), (9, 6), (10, 6), (11, 6), (12, 6),                    (17, 6),          (19, 6),          (21, 6), (22, 6),
                            (2, 5), (3, 5), (4, 5), (5, 5),                           (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),
                            (2, 4), (3, 4), (4, 4), (5, 4),                           (8, 4), (9, 4), (10, 4), (11, 4), (12, 4),
                            (2, 3), (3, 3), (4, 3),                                           (9, 3), (10, 3), (11, 3), (12, 3),                    (17, 3), (18, 3), (19, 3),          (21, 3), (22, 3),
                            (2, 2), (3, 2), (4, 2), (5, 2),                           (8, 2), (9, 2), (10, 2), (11, 2), (12, 2),                    (17, 2), (18, 2), (19, 2), (20, 2), (21, 2), (22, 2)],

        parkingSpotsPos=[(10, 21), (2, 20), (12, 19), (7, 18), (8, 15), (4, 13), (12, 13), (2, 6), (5, 3), (8, 3), (18, 20), (17, 13), (18, 6), (21, 19), (22, 14), (20, 6), (20, 3)],
        roundAboutPos = [(14, 10), (15, 10), (14, 9), (15, 9)],
        trafficLightPos = [(15, 21), (16, 21), (5, 15), (6, 15), (0, 12), (1, 12), (23, 7), (24, 7), (13, 2), (14, 2), (15, 3), (16, 3), (17, 23), (17, 22), (8, 17), (8, 16), (2, 11), (2, 10), (22, 9), (22, 8), (17, 5), (17, 4), (12, 1), (12, 0)],
    ):
        """
        Crea un nuevo modelo Flockers.

        Args:
            num: Número de Boids  # Cambia population a num
            width, height: Tamaño del espacio.
            speed: Qué tan rápido deben moverse los Boids.
            vision: Hasta dónde debería mirar cada Boid en busca de sus vecinos.
            separation: Cuál es la distancia mínima que cada Boid intentará
                    mantener con cualquier otro.
            cohere, separate, match: factores para la importancia relativa de
                    los tres impulsos.
        """
        self.population = num
        self.vision = vision
        self.speed = speed
        self.separation = separation
        self.pathIndex = pathIndex
        
        self.buildingPos = buildingPos if buildingPos else []
        self.parkingSpotsPos = parkingSpotsPos if parkingSpotsPos else []
        self.roundAboutPos = roundAboutPos if roundAboutPos else []
        self.trafficLightPos = trafficLightPos if trafficLightPos else []

        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(width, height, True)
        self.factors = {"cohere": cohere, "separate": separate, "match": match}
        self.make_agents()
        self.running = True

    def make_agents(self):
        """
        Crea agentes self.population, con posiciones y direcciones iniciales aleatorias.
        """
        # CARS
        for i in range(self.population):
            r = self.random.randint(0, len(self.parkingSpotsPos)-1)
            rg = self.random.randint(0, len(self.parkingSpotsPos)-1)
            if r == rg:
                r = self.random.randint(0, len(self.parkingSpotsPos)-1)
                rg = self.random.randint(0, len(self.parkingSpotsPos)-1)
            else:
                print("Valor de r: ", r)
                print("Valor de rg: ", rg)

                x = self.parkingSpotsPos[r][0]
                y = self.parkingSpotsPos[r][1]

                xg = self.parkingSpotsPos[rg][0]
                yg = self.parkingSpotsPos[rg][1]
                pos = np.array((x, y))
                path = []
                goal = np.array((xg, yg))

                car = Car(
                    i,
                    self,
                    pos,
                    path,
                    self.pathIndex,
                    goal,
                )

                self.space.place_agent(car, pos)
                self.schedule.add(car)
        
        # BUILDINGS
        for i in range(len(self.buildingPos)):
            x = self.buildingPos[i][0]
            y = self.buildingPos[i][1]

            pos = np.array((x, y))

            building = Buildings(
                i + self.population,
                self,
                pos,
            )
                
            self.space.place_agent(building, pos)
            self.schedule.add(building)

        # ROUNDABOUT
        for i in range(len(self.roundAboutPos)):
            x = self.roundAboutPos[i][0]
            y = self.roundAboutPos[i][1]

            pos = np.array((x, y))

            roundabout = RoundAbout(
                i + self.population + len(self.buildingPos),
                self,
                pos,
            )
                
            self.space.place_agent(roundabout, pos)
            self.schedule.add(roundabout)

        # TRAFFIC LIGHTS
        for i in range(len(self.trafficLightPos)):
            x = self.trafficLightPos[i][0]
            y = self.trafficLightPos[i][1]

            pos = np.array((x, y))

            if i == 0 | i == 1 | i ==12 | i == 13 | i == 14 | i == 15 | i == 16 | i == 17 | i == 18 | i == 19 |i == 22 | i == 23:
                state = "green"
            else:
                state = "red"

            trafficlight = TrafficLight(
                i + self.population + len(self.buildingPos) + len(self.roundAboutPos),
                self,
                pos,
                state,
            )
                
            self.space.place_agent(trafficlight, pos)
            self.schedule.add(trafficlight)

         
        # STREET 
        #for i in range(len(self.streetPos)-1):
        #    x = self.streetPos[i][0]
        #    y = self.streetPos[i][1]

        #    pos = np.array((x, y))
        #    direction = "up"

        #   street = Street(
        #      i + self.population + len(self.buildingPos) + len(self.roundAboutPos) + len(self.trafficLightPos),
        #        self,
        #        pos,
        #        direction,
        #    )
                
        #    self.space.place_agent(street, pos)
        #    self.schedule.add(street)

    def step(self):
        self.schedule.step()

    def getCarPositions(self):
        car_positions = []
        for agent in self.schedule.agents:
            if isinstance(agent, Car):
                car_positions.append(agent.pos.tolist())
        return car_positions

    def getBuildingsPosition(self):
        building_positions = []
        for agent in self.schedule.agents:
            if isinstance(agent, Buildings):
                building_positions.append(agent.pos.tolist())
        return building_positions

    def getRoundAboutPositions(self):
        roundabout_positions = []
        for agent in self.schedule.agents:
            if isinstance(agent, RoundAbout):
                roundabout_positions.append(agent.pos.tolist())
        return roundabout_positions

    def getTrafficLightPositions(self):
        trafficlight_positions = []
        for agent in self.schedule.agents:
            if isinstance(agent, TrafficLight):
                trafficlight_positions.append(agent.pos.tolist())
        return trafficlight_positions

    #def getStreetPositions(self):
    #    street_positions = []
    #    for agent in self.schedule.agents:
    #        if isinstance(agent, Street):
    #            street_positions.append(agent.pos.tolist())
    #    return street_positions


