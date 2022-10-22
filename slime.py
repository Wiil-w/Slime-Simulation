import numpy as np
import cv2 as cv


class Simulation:

    # delay between frames in ms
    DELAY: int = 1
    RENDER_SPEED: int = 1
    SPEED: int = 5
    MAP_SIZE: int = 500
    AGENTS_NUM: int = 50
    CIRCLE_SIZE: float = 0.4
    SIZE: int = 3
    TRAIL_STRENGTH: float = 25
    AGENT_FOV: float = np.pi / 6
    VISION_DISTANCE: int = 2 * SPEED

    # How heavy the blur is on the map
    # Which is used to spread out trails
    BLUR_STRENTH: int = 11

    # How fast trails fade to 0
    # Trails are multipled by this value every step
    DISSIPATE_TARGET: int = AGENTS_NUM * 400

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
        self.update_agents()
        self.lay_trails()
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
        np.apply_along_axis(self.rotate, 0, self.agents[:, 2])

        # rad_straight = self.agents[:, 2]
        # straight = self.get_vision(
        #     np.sin(self.agents[:, 2]),
        #     np.cos(self.agents[:, 2]),
        # )

        # rad_right = rad_straight - self.AGENT_FOV
        # right = self.get_vision(
        #     np.sin(rad_right),
        #     np.cos(rad_right),
        # )

        # rad_left = rad_straight + self.AGENT_FOV
        # left = self.get_vision(
        #     np.sin(rad_left),
        #     np.cos(rad_left),
        # )

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

    def rotate(
        self,
        rad: np.float64,
    ) -> np.float64:

        # straight
        h = self.VISION_DISTANCE * np.sin(rad)
        w = self.VISION_DISTANCE * np.cos(rad)
        trail_sum = self.trails_map[h - 3: h + 3, w - 3: w + 3].sum()
        result = rad

        # left
        rad_left = rad + self.AGENT_FOV
        h = self.VISION_DISTANCE * np.sin(rad_left)
        w = self.VISION_DISTANCE * np.cos(rad_left)
        val = self.trails_map[h - 3: h + 3, w - 3: w + 3].sum()
        if val > trail_sum:
            trail_sum = val
            result = rad_left

        # right
        rad_right = rad - self.AGENT_FOV
        h = self.VISION_DISTANCE * np.sin(rad_right)
        w = self.VISION_DISTANCE * np.cos(rad_right)
        trail_sum = self.trails_map[h - 3: h + 3, w - 3: w + 3].sum()
        if val > trail_sum:
            trail_sum = val
            result = rad_right

        return result

    def lay_trails(self) -> None:
        h, w = self.agents[:, :2].astype(int).T
        self.trails_map[h, w] += self.TRAIL_STRENGTH

    def dissipate_trails(self):
        self.trails_map = cv.GaussianBlur(
            self.trails_map, (self.BLUR_STRENTH, self.BLUR_STRENTH), 0
        )

        self.trails_map = np.clip(self.trails_map, 0, 20)

        while self.trails_map.sum() > self.DISSIPATE_TARGET:
            self.trails_map *= 0.95

    def draw_map(self) -> None:
        self.frame = np.zeros((self.MAP_SIZE, self.MAP_SIZE))

        # Draw trails
        self.frame += self.trails_map


if "__main__" == __name__:
    s = Simulation()
    s.update()
