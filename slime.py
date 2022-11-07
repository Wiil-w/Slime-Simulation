import glob
import numpy as np
import numpy.typing as npt
import cv2 as cv
from PIL import Image as img


class Simulation:

    delay: int = 1
    render_speed: int = 1
    speed: int = 1

    screen_size: int = 3
    zoom: int = 4 - 2 ** (screen_size - 1)
    map_width: int = 320 * 2 ** (screen_size - 1)
    map_height: int = 180 * 2 ** (screen_size - 1)
    circle_size: float = 0.7

    agents_num: int = 20000
    agent_fov: float = np.pi / 8
    agent_turn: float = agent_fov / 2
    agent_deviation: float = 0.25
    vision_distance: int = speed * 8
    vision_size: int = 3

    trail_strength: float = 0.8
    blur_size: int = 3
    dissipation_rate: float = 0.05
    decay_rate: float = 0.8

    # 0 - realtime; 1 - gif
    mode: int = 1
    gif_iterations: int = 10000
    path: str = "results"

    def __init__(self) -> None:
        self.tick = 0
        self.trails_map = np.zeros(
            (self.map_height, self.map_width), dtype=np.float64)
        self.create_agents()

        vision_area = self.vision_size**2
        offset = self.vision_size // 2

        self.mat_h = np.full(
            (self.agents_num, vision_area),
            (np.arange(vision_area) // self.vision_size) - offset,
        )

        self.mat_w = np.full(
            (self.agents_num, vision_area),
            (np.arange(vision_area) % self.vision_size) - offset,
        )

    def update(self) -> None:
        """runs the simulation"""

        self.render_map = np.full(
            (self.map_height, self.map_width, 3), [0, 0, 0], dtype=np.uint8
        )

        while True:
            self.tick += 1
            self.update_simulation()
            if not (self.tick % self.render_speed):
                self.render()

            k = cv.waitKey(self.delay)
            if k == 113 or (self.mode and self.tick//self.render_speed >= self.gif_iterations):
                break

        if self.mode == 1:
            # generate gif from rendered images
            images = glob.glob(f"{self.path}/*.png")
            images.sort()

            frames = [img.open(image) for image in images]
            frame_one = frames[0]
            frame_one.save(f"{self.path}/out.gif", format="GIF", append_images=frames,
                           save_all=True, duration=20, loop=0)

    def render(self) -> None:
        self.render_map[:, :, 0] = (self.trails_map / self.trail_strength).astype(
            np.uint8
        )
        self.render_map[:, :, 1] = self.render_map[:, :, 0] // 2

        if self.zoom > 1:
            render = cv.resize(
                self.render_map,
                (self.map_width * self.zoom, self.map_height * self.zoom),
                interpolation=cv.INTER_NEAREST,
            )
        else:
            render = self.render_map

        match self.mode:
            case 0:  # realtime
                cv.imshow("Simulation", render)

            case 1:  # into images
                im = img.fromarray(cv.cvtColor(render, cv.COLOR_BGR2RGB))
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(f"{self.path}/{self.tick//self.render_speed}.png")

    def update_simulation(self) -> None:
        self.lay_trails()
        self.update_agents()
        self.dissipate_trails()

    def create_agents(self, start_shape=0) -> None:
        """create_agents creates a collection of agents in a circle"""

        match start_shape:
            case 0:  # circle
                radius = int(
                    self.circle_size * min(self.map_height, self.map_width) / 2
                )

                t = radius * \
                    np.sqrt(np.random.uniform(0, 1, size=self.agents_num))
                u = np.random.uniform(0, 1, size=self.agents_num) * 2 * np.pi

                self.agents_angle = -u  # - np.pi/2
                self.agents_pos = np.column_stack(
                    [
                        t * np.cos(u) + self.map_height / 2,
                        t * np.sin(u) + self.map_width / 2,
                    ]
                )

            case 1:  # point in middle
                # single point in middle (WIP)
                self.agents_angle = np.linspace(
                    0, 1, num=self.agents_num) * 2 * np.pi
                self.agents_pos = np.full(
                    (self.agents_num, 2), (self.map_height / 2, self.map_width / 2)
                )

    def update_agents(self) -> None:

        # rotate agent direction (vector)
        self.rotate_angle()

        # update agents position
        h = self.agents_pos[:, 0] + self.speed * np.sin(self.agents_angle)
        w = self.agents_pos[:, 1] + self.speed * np.cos(self.agents_angle)

        x_out_of_bounds = (w < 0) | (w >= self.map_width - 1)
        out_of_bounds = (h < 0) | (h >= self.map_height - 1) | x_out_of_bounds

        self.agents_pos[:, 0] = np.clip(h, 0, self.map_height - 1)
        self.agents_pos[:, 1] = np.clip(w, 0, self.map_width - 1)

        # update agents directions (radians)
        # bouncing off horizontal walls means inverting the radians
        # bouncing off vertical walls means subtracting the radians from pi (invert and add pi)
        self.agents_angle[out_of_bounds] *= -1
        self.agents_angle[out_of_bounds] += (
            2 * self.agent_turn * (np.random.rand(out_of_bounds.sum()) - 0.5)
        )
        self.agents_angle[x_out_of_bounds] += np.pi

    def rotate_angle(self) -> None:
        scent_straight = self.calc_rotation(self.agents_angle)
        scent_right = self.calc_rotation(self.agents_angle - self.agent_fov)
        scent_left = self.calc_rotation(self.agents_angle + self.agent_fov)

        direction: np.uint8 = (scent_straight < scent_right) + (
            scent_straight < scent_left
        ) * 2

        turn_strength = (
            np.random.rand(self.agents_num) * (2 * self.agent_deviation)
            + (1 - self.agent_deviation)
        ) * self.agent_turn

        # # don't turn if the scent is stronger straight ahead
        # self.agents_rad[direction == 0] += 0

        # turn right if scent is stronger to the right
        right = direction == 1
        self.agents_angle[right] -= turn_strength[right]

        # turn left if scent is stronger to the left
        left = direction == 2
        self.agents_angle[left] += turn_strength[left]

        # turn randomly if scent is stronger in both drections than in front
        equal = direction == 3
        self.agents_angle[equal] += (
            np.random.randint(-2, 3, equal.sum())
        ) * self.agent_turn
        # self.agents_rad[equal] += turn_strength[equal] * (np.random.randint(2, equal.sum())*2-1)

    def calc_rotation(self, rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        h = self.agents_pos[:, 0] + self.vision_distance * np.sin(rad)
        w = self.agents_pos[:, 1] + self.vision_distance * np.cos(rad)

        h = self.mat_h + h.reshape(-1, 1)
        w = self.mat_w + w.reshape(-1, 1)

        in_bounds = (0 <= h) & (h < self.map_height) \
            & (0 <= w) & (w < self.map_width)

        h: npt.NDArray[np.uint] = np.clip(h, 0, self.map_height - 1).astype(int)
        w: npt.NDArray[np.uint] = np.clip(w, 0, self.map_width - 1).astype(int)

        return (self.trails_map[h, w] * in_bounds).sum(axis=1)

    def lay_trails(self) -> None:
        h, w = self.agents_pos.astype(int).T
        self.trails_map[h, w] = 255 * self.trail_strength

    def dissipate_trails(self) -> None:
        # blur the trails map
        blur_map: npt.NDArray[np.float64] = cv.blur(
            self.trails_map, (self.blur_size, self.blur_size)
        )
        blur_map[blur_map < 0] = 0

        # linear interpolation between the blurred map and the original map
        self.trails_map: npt.NDArray[np.float64] = (
            self.dissipation_rate * blur_map
            + (1 - self.dissipation_rate) * self.trails_map
        )

        # decay the trails
        self.trails_map[self.trails_map >= self.decay_rate] -= self.decay_rate


if "__main__" == __name__:
    s = Simulation()
    s.update()
