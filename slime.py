import numpy as np
import cv2 as cv


class Simulation:

    # delay between frames in ms
    DELAY: int = 15
    RENDER_SPEED: int = 1
    MAP_SIZE: int = 100
    AGENTS_NUM: int = 5
    CIRCLE_SIZE: float = 0.4
    SIZE: int = 10

    def __init__(self) -> None:
        self.tick = 0
        self.trails_map = np.zeros((self.MAP_SIZE, self.MAP_SIZE))
        self.create_agents()

    def update(self) -> None:
        """runs the simulation"""
        while True:
            self.tick += 1
            self.update_agents()
            self.draw_map()

            if not (self.tick % self.RENDER_SPEED):
                cv.imshow("Simulation", self.frame)

            k = cv.waitKey(self.DELAY)
            if k == 113:
                break

    def create_agents(self) -> None:
        """create_agents creates a collection of agents in a circle"""

        radius = int(self.CIRCLE_SIZE * self.MAP_SIZE)

        # Circle
        # theta = np.linspace(0, 2 * np.pi, self.AGENTS_NUM)
        # a = radius * np.cos(theta)
        # b = radius * np.sin(theta)

        t = np.random.uniform(0, 1, size=self.AGENTS_NUM)
        u = np.random.uniform(0, 1, size=self.AGENTS_NUM)

        x = radius * np.sqrt(t) * np.cos(2 * np.pi * u) + self.MAP_SIZE / 2
        y = radius * np.sqrt(t) * np.sin(2 * np.pi * u) + self.MAP_SIZE / 2

        self.agents = (
            np.column_stack(
                [
                    x,
                    y,
                    (np.random.rand(self.AGENTS_NUM) - 0.5) * 10,  # vx
                    (np.random.rand(self.AGENTS_NUM) - 0.5) * 10,  # vy
                ]
            )
            + np.random.rand(self.AGENTS_NUM, 4) * 0.1
        )

    def update_agents(self) -> None:
        self.agents[:, :2] += self.agents[:, 2:]

        in_bounds = (0 <= self.agents[:, :2]) & (
            self.agents[:, :2] < self.MAP_SIZE)

        self.agents[:, :2][~in_bounds] = np.clip(
            self.agents[:, :2][~in_bounds], 0, self.MAP_SIZE - 1)

        self.agents[:, 2:][~in_bounds] *= -1

    def draw_map(self) -> None:
        self.frame = np.zeros((self.MAP_SIZE, self.MAP_SIZE))

        # Draw agents
        h, w = self.agents[:, :2].astype(int).T
        self.frame[h, w] = 255

        # Scale the map
        h, w = self.frame.shape
        self.frame = cv.resize(
            self.frame, (w*self.SIZE, h*self.SIZE), interpolation=cv.INTER_NEAREST)


if "__main__" == __name__:
    s = Simulation()
    s.update()
