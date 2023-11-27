import mesa
import numpy as np
import random

from agent import *

class BoidFlockers(mesa.Model):
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
        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(width, height, True)
        self.factors = {"cohere": cohere, "separate": separate, "match": match}
        self.make_agents()
        self.running = True

    def make_agents(self):
        """
        Crea agentes self.population, con posiciones y direcciones iniciales aleatorias.
        """
        parkingSlots = [(3, 7), (3, 21), (5, 14), (6, 4), (7, 19), (9, 4), (9, 16), (10, 22), (12, 14), (12, 20), (17, 14), (18, 7), (18, 21), (20, 4), (20, 7), (21, 20), (22, 15)]

        for i in range(self.population):
            r = self.random.randint(0, len(parkingSlots)-1)
            x = parkingSlots[r][0]
            y =parkingSlots[r][1]
            pos = np.array((x, y))
            print(pos)
            velocity = np.random.random(2) * 2 - 1
            boid = Boid(
                i,
                self,
                pos,
                self.speed,
                velocity,
                self.vision,
                self.separation,
                **self.factors,
            )
            self.space.place_agent(boid, pos)
            self.schedule.add(boid)

    def step(self):
        self.schedule.step()

    def get_positions(self):
        places = []
        for agent in self.schedule.agents:
            places.append(agent.pos.tolist())  # Convierte la posición a lista antes de agregarla
        return places


