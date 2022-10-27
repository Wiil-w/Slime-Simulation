import numpy as np
import numpy.typing as npt
import cv2 as cv


class Simulation:

    delay: int = 1
    render_speed: int = 1
    speed: int = 1

    screen_size: int = 3
    zoom: int = 4 - 2**(screen_size-1)
    map_width: int = 320 * 2**(screen_size-1)
    map_height: int = 180 * 2**(screen_size-1)
    circle_size: float = 0.7

    agents_num: int = 20000
    agent_fov: float = np.pi / 8
    agent_turn: float = agent_fov / 2
    agent_deviation: float = .25
    vision_distance: int = speed * 8
    vision_size: int = 3

    trail_strength: float = .8
    blur_size: int = 3
    dissipation_rate: float = .05
    decay_rate: float = .8

    def __init__(self) -> None:
        self.tick = 0
        self.trails_map = np.zeros((self.map_height, self.map_width))
        self.create_agents()

        vision_area = self.vision_size**2
        offset = self.vision_size//2

        self.mat_h = np.full((self.agents_num, vision_area), 
                             (np.arange(vision_area) // self.vision_size) - offset)

        self.mat_w = np.full((self.agents_num, vision_area),
                             (np.arange(vision_area) % self.vision_size) - offset)

    def update(self) -> None:
        """runs the simulation"""
        while True:
            self.tick += 1
            self.update_simulation()

            if not (self.tick % self.render_speed):
                trails = (self.trails_map /
                          self.trail_strength).astype(np.uint8)
                          
                if self.zoom > 1:
                    trails = cv.resize(trails,
                                       (self.map_width * self.zoom,
                                        self.map_height * self.zoom),
                                       interpolation=cv.INTER_NEAREST)

                cv.imshow("Simulation", trails)

            k = cv.waitKey(self.delay)
            if k == 113:
                break

    def update_simulation(self) -> None:
        self.lay_trails()
        self.update_agents()
        self.dissipate_trails()

    def create_agents(self) -> None:
        """create_agents creates a collection of agents in a circle"""

        radius = int(self.circle_size *
                     min(self.map_height, self.map_width) / 2)

        t = radius * np.sqrt(np.random.uniform(0, 1, size=self.agents_num))
        u = np.random.uniform(0, 1, size=self.agents_num) * 2 * np.pi

        self.agents_rad = -u  # - np.pi/2
        self.agents_pos = np.column_stack(
            [
                t * np.cos(u) + self.map_height / 2,
                t * np.sin(u) + self.map_width / 2,
            ]
        )

        # single point in middle (WIP)
        # self.agents_rad = np.linspace(0, 1, num=self.agents_num) * 2 * np.pi
        # self.agents_pos = np.full((self.agents_num, 2), (self.map_height / 2, self.map_width / 2))

    def update_agents(self) -> None:

        # rotate agent direction (vector)
        self.rotate()

        # update agents position
        h = self.agents_pos[:, 0] + self.speed * np.sin(self.agents_rad)
        w = self.agents_pos[:, 1] + self.speed * np.cos(self.agents_rad)

        x_out_of_bounds = (w < 0) | (w >= self.map_width-1)
        out_of_bounds = (h < 0) | (h >= self.map_height - 1) | x_out_of_bounds

        self.agents_pos[:, 0] = np.clip(h, 0, self.map_height - 1)
        self.agents_pos[:, 1] = np.clip(w, 0, self.map_width - 1)

        # update agents directions (radians)
        # bouncing off horizontal walls means inverting the radians
        # bouncing off vertical walls means subtracting the radians from pi (invert and add pi)
        self.agents_rad[out_of_bounds] *= -1
        self.agents_rad[out_of_bounds] += 2 * self.agent_turn * \
            (np.random.rand(out_of_bounds.sum()) - .5)
        self.agents_rad[x_out_of_bounds] += np.pi

    def rotate(self) -> None:
        scent_straight = self.calc_rotation(self.agents_rad)
        scent_right = self.calc_rotation(self.agents_rad - self.agent_fov)
        scent_left = self.calc_rotation(self.agents_rad + self.agent_fov)

        direction = (scent_straight <= scent_right) + \
            (scent_straight <= scent_left) * 2

        turn_strength = (np.random.rand(self.agents_num) *
                         (2*self.agent_deviation) + (1-self.agent_deviation)) * self.agent_turn

        # # don't turn if the scent is stronger straight ahead
        # self.agents_rad[direction == 0] += 0

        # turn right if scent is stronger to the right
        right = direction == 1
        self.agents_rad[right] -= turn_strength[right]

        # turn left if scent is stronger to the left
        left = direction == 2
        self.agents_rad[left] += turn_strength[left]

        # turn randomly if scent is stronger in both drections than in front
        equal = direction == 3
        self.agents_rad[equal] += (np.random.randint(-2,
                                   3, equal.sum())) * self.agent_turn
        # self.agents_rad[equal] += turn_strength[equal] * (np.random.randint(2, equal.sum())*2-1)

    def calc_rotation(self, rad: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:

        h = self.agents_pos[:, 0] + self.vision_distance * np.sin(rad)
        w = self.agents_pos[:, 1] + self.vision_distance * np.cos(rad)

        h = self.mat_h + h.reshape(-1, 1)
        w = self.mat_w + w.reshape(-1, 1)

        in_bounds = (0 <= h) & (h < self.map_height) \
            & (0 <= w) & (w < self.map_width)

        h = np.clip(h, 0, self.map_height - 1).astype(int)
        w = np.clip(w, 0, self.map_width - 1).astype(int)

        return (self.trails_map[h, w] * in_bounds).sum(axis=1)

    def lay_trails(self) -> None:
        h, w = self.agents_pos.astype(int).T
        self.trails_map[h, w] = 255 * self.trail_strength

    def dissipate_trails(self) -> None:
        # blur the trails map
        blur_map = cv.blur(
            self.trails_map, (self.blur_size, self.blur_size)
        )

        # linear interpolation between the blurred map and the original map
        self.trails_map = self.dissipation_rate * blur_map + \
            (1 - self.dissipation_rate) * self.trails_map

        # decay the trails
        self.trails_map[self.trails_map >= self.decay_rate] -= self.decay_rate


if "__main__" == __name__:
    s = Simulation()
    s.update()
