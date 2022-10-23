import numpy as np
import cv2 as cv


class Simulation:

    # delay between frames in ms
    DELAY: int = 1
    RENDER_SPEED: int = 1
    SPEED: int = 2
    MAP_SIZE: int = 500
    AGENTS_NUM: int = 500
    CIRCLE_SIZE: float = 0.4
    TRAIL_STRENGTH: float = 15
    AGENT_FOV: float = np.pi / 12
    VISION_DISTANCE: int = SPEED * 2

    # How heavy the blur is on the map
    # Which is used to spread out trails
    BLUR_STRENTH: int = 3

    # How fast trails fade to 0
    # Trails are multipled by this value every step
    DISSIPATE_TARGET: int = AGENTS_NUM * 100

    def __init__(self) -> None:
        self.tick = 0
        self.trails_map = np.zeros((self.MAP_SIZE, self.MAP_SIZE))
        self.create_agents()

    def update(self) -> None:
        """runs the simulation"""
        while True:
            self.tick += 1
            self.update_simulation()
            self.draw_map()

            if not (self.tick % self.RENDER_SPEED):
                cv.imshow("Simulation", self.frame)

            k = cv.waitKey(self.DELAY)
            if k == 113:
                break

    def update_simulation(self) -> None:
        self.lay_trails()
        self.update_agents()
        self.dissipate_trails()

    def create_agents(self) -> None:
        """create_agents creates a collection of agents in a circle"""

        radius = int(self.CIRCLE_SIZE * self.MAP_SIZE)

        t = np.random.uniform(0, 1, size=self.AGENTS_NUM)
        u = np.random.uniform(0, 1, size=self.AGENTS_NUM)

        x = radius * np.sqrt(t) * np.cos(2 * np.pi * u) + self.MAP_SIZE / 2
        y = radius * np.sqrt(t) * np.sin(2 * np.pi * u) + self.MAP_SIZE / 2
        r = np.random.rand(self.AGENTS_NUM) * 2 * np.pi

        self.agents = np.column_stack(
            [
                x,
                y,
                r,
            ]
        )

    def update_agents(self) -> None:

        # rotate agent direction (vector)
        self.rotate()

        # update agents position
        self.agents[:, 0] += self.SPEED * np.sin(self.agents[:, 2])
        self.agents[:, 1] += self.SPEED * np.cos(self.agents[:, 2])

        in_bounds = (0 <= self.agents[:, :2]) & (
            self.agents[:, :2] < self.MAP_SIZE)

        # keep agents in bounds
        self.agents[:, :2][~in_bounds] = np.clip(
            self.agents[:, :2][~in_bounds], 0, self.MAP_SIZE - 1
        )

        # update agents directions (radians)
        # bouncing off horizontal walls means inverting the radians
        # bouncing off vertical walls means subtracting the radians from pi (invert and add pi)
        self.agents[:, 2][~in_bounds.all(axis=1)] *= -1
        self.agents[:, 2][~in_bounds[:, 1]] += np.pi

    # def rotate(self) -> None:
    #     h, w = self.agents[:, :2].astype(int).T
    #     trail_sum = np.zeros((self.AGENTS_NUM))
    #     result = self.agents[:, 2]
    #     for i in range(-1, 2):
    #         rad_i = self.agents[:, 2] + self.AGENT_FOV * i
    #         h_v = np.clip(h + self.VISION_DISTANCE * np.sin(rad_i), 0, self.MAP_SIZE - 1).astype(int)
    #         w_v = np.clip(w + self.VISION_DISTANCE * np.cos(rad_i), 0, self.MAP_SIZE - 1).astype(int)
    #         val = self.trails_map[h_v, w_v]

    #         cond = val > trail_sum
    #         trail_sum = np.where(cond, trail_sum, val)
    #         result = np.where(cond, result, rad_i)

    #     self.agents[:, 2] = result

    def rotate(self) -> None:

        trail_sum = self.calc_rotation(self.agents[:, 2])
        result = self.agents[:, 2]

        rad_left = self.agents[:, 2] + self.AGENT_FOV
        val = self.calc_rotation(rad_left)
        cond = val > trail_sum
        trail_sum = np.where(cond, val, trail_sum)
        result = np.where(cond, rad_left, result)

        rad_right = self.agents[:, 2] - self.AGENT_FOV
        val = self.calc_rotation(rad_right)
        cond = val > trail_sum
        trail_sum = np.where(cond, val, trail_sum)
        result = np.where(cond, rad_right, result)

        self.agents[:, 2] = result
        
        cond = np.random.rand(self.AGENTS_NUM) < .1
        self.agents[cond, 2] += np.where(np.random.randint(
            2, size=cond.sum()), self.AGENT_FOV, -self.AGENT_FOV)

    def calc_rotation(self, rad: np.ndarray[any, np.floating]) -> np.ndarray[any, np.int32]:
        h, w = self.agents[:, :2].astype(int).T
        h_v = np.clip(h + self.VISION_DISTANCE * np.sin(rad),
                      0, self.MAP_SIZE - 1).astype(int)
        w_v = np.clip(w + self.VISION_DISTANCE * np.cos(rad),
                      0, self.MAP_SIZE - 1).astype(int)
        return self.trails_map[h_v, w_v]

    def lay_trails(self) -> None:
        h, w = self.agents[:, :2].astype(int).T
        self.trails_map[self.trails_map > 3] -= 3
        self.trails_map[h, w] += self.TRAIL_STRENGTH

    def dissipate_trails(self):
        self.trails_map = cv.GaussianBlur(
            self.trails_map, (self.BLUR_STRENTH, self.BLUR_STRENTH), 0
        )

        self.trails_map = np.clip(self.trails_map, 0, self.TRAIL_STRENGTH)

        while self.trails_map.sum() > self.DISSIPATE_TARGET:
            self.trails_map *= 0.95

    def draw_map(self) -> None:
        self.frame = np.zeros((self.MAP_SIZE, self.MAP_SIZE))

        # Draw trails
        self.frame += self.trails_map


if "__main__" == __name__:
    s = Simulation()
    s.update()
