import numpy as np
import cv2 as cv


class Simulation:

    # delay between frames in ms
    delay: int = 1
    render_speed: int = 1
    speed: int = 1
    map_size: int = 500
    agents_num: int = 50
    circle_size: float = 0.4
    trail_strength: float = 1
    agent_fov: float = np.pi / 12
    vision_distance: int = speed * 2

    # How heavy the blur is on the map
    # Which is used to spread out trails
    blur_strenth: int = 3
    dissipation_rate: float = .95

    def __init__(self) -> None:
        self.tick = 0
        self.trails_map = np.zeros((self.map_size, self.map_size))
        self.create_agents()

    def update(self) -> None:
        """runs the simulation"""
        while True:
            self.tick += 1
            self.update_simulation()
            self.draw_map()

            if not (self.tick % self.render_speed):
                cv.imshow("Simulation", self.frame)

            k = cv.waitKey(self.delay)
            if k == 113:
                break

    def update_simulation(self) -> None:
        self.lay_trails()
        self.update_agents()
        self.dissipate_trails()

    def create_agents(self) -> None:
        """create_agents creates a collection of agents in a circle"""

        radius = int(self.circle_size * self.map_size)

        t = np.random.uniform(0, 1, size=self.agents_num)
        u = np.random.uniform(0, 1, size=self.agents_num) * 2 * np.pi

        self.agents_rad = np.random.rand(self.agents_num) * 2 * np.pi
        self.agents_pos = np.column_stack(
            [
                radius * np.sqrt(t) * np.cos(u) + self.map_size / 2,
                radius * np.sqrt(t) * np.sin(u) + self.map_size / 2,
            ]
        )

    def update_agents(self) -> None:

        # rotate agent direction (vector)
        self.rotate()

        # update agents position
        self.agents_pos[:, 0] += self.speed * np.sin(self.agents_rad)
        self.agents_pos[:, 1] += self.speed * np.cos(self.agents_rad)

        in_bounds = (0 <= self.agents_pos) & (
            self.agents_pos < self.map_size)

        # keep agents in bounds
        self.agents_pos[~in_bounds] = np.clip(
            self.agents_pos[~in_bounds], 0, self.map_size - 1
        )

        # update agents directions (radians)
        # bouncing off horizontal walls means inverting the radians
        # bouncing off vertical walls means subtracting the radians from pi (invert and add pi)
        self.agents_rad[~in_bounds.all(axis=1)] *= -1
        self.agents_rad[~in_bounds[:, 1]] += np.pi

    def rotate(self) -> None:

        # rads = np.column_stack(
        #     [
        #         self.agents_rad - self.agent_fov,
        #         self.agents_rad,
        #         self.agents_rad + self.agent_fov,
        #     ]
        # )

        trail_sum = self.calc_rotation(self.agents_rad)
        result = self.agents_rad

        rad_left = self.agents_rad + self.agent_fov
        val = self.calc_rotation(rad_left)
        cond = val > trail_sum
        trail_sum = np.where(cond, val, trail_sum)
        result = np.where(cond, rad_left, result)

        rad_right = self.agents_rad - self.agent_fov
        val = self.calc_rotation(rad_right)
        cond = val > trail_sum
        trail_sum = np.where(cond, val, trail_sum)
        result = np.where(cond, rad_right, result)

        self.agents_rad = result

        cond = np.random.rand(self.agents_num) < .1
        self.agents_rad[cond] += np.where(np.random.randint(
            2, size=cond.sum()), self.agent_fov, -self.agent_fov)

    def calc_rotation(self, rad: np.ndarray[any, np.floating]) -> np.ndarray[any, np.int32]:
        h_v = np.clip(self.agents_pos[:, 0] + self.vision_distance * np.sin(rad),
                      0, self.map_size - 1).astype(int)
        w_v = np.clip(self.agents_pos[:, 1] + self.vision_distance * np.cos(rad),
                      0, self.map_size - 1).astype(int)

        return self.trails_map[h_v, w_v]

    def lay_trails(self) -> None:
        h, w = self.agents_pos.astype(int).T
        self.trails_map[h, w] = 255 * self.trail_strength

    def dissipate_trails(self):

        # blur the trails map
        blur_map = cv.blur(
            self.trails_map, (self.blur_strenth, self.blur_strenth)
        )

        # linear interpolation between the blurred map and the original map
        self.trails_map = self.dissipation_rate * self.trails_map + (
            1 - self.dissipation_rate
        ) * blur_map
        
        # decay the trails
        self.trails_map[self.trails_map != 0] -= 1
        # self.trails_map *= self.dissipation_rate

    def draw_map(self) -> None:
        self.frame = np.zeros((self.map_size, self.map_size), dtype=np.uint8) + self.trails_map.astype(np.uint8)


if "__main__" == __name__:
    s = Simulation()
    s.update()
