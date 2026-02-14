"""
Smart Traffic Light Simulation - Pygame visualization of adaptive traffic signals.
"""
import random
import time

import pygame

# Constants
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600
LANES = ['R1', 'R2', 'R3', 'R4']
COLORS = {
    'red': (255, 70, 70),
    'green': (0, 200, 100),
    'car': (100, 149, 237),
    'bg': (24, 24, 24),
    'white': (255, 255, 255),
    'lane_box': (50, 50, 50),
}
MAX_CARS_PER_LANE = 8
MIN_GREEN_TIME = 10
WAIT_LIMIT = 12
VEHICLE_DIFF_THRESHOLD = 5
R_THRESHOLDS = {lane: 3 for lane in LANES}
MOVE_CARS_INTERVAL = 30  # frames (~1 sec at 30fps)
ADD_VEHICLES_INTERVAL = 15  # frames (~0.5 sec at 30fps)

# Lane setup
LANE_RECTS = {
    'R1': pygame.Rect(80, 150, 150, 300),
    'R2': pygame.Rect(260, 150, 150, 300),
    'R3': pygame.Rect(440, 150, 150, 300),
    'R4': pygame.Rect(620, 150, 150, 300),
}


def main():
    pygame.init()
    font = pygame.font.SysFont('Arial', 20)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Smart Traffic Light Simulation')

    vehicle_queues = {lane: [] for lane in LANES}
    last_green_time = {lane: time.time() - WAIT_LIMIT for lane in LANES}
    current_signals = {lane: '🔴' for lane in LANES}
    current_green_lane = 'R1'
    green_start_time = time.time()
    frame_count = 0

    def draw_lanes():
        screen.fill(COLORS['bg'])
        for lane in LANES:
            rect = LANE_RECTS[lane]
            pygame.draw.rect(screen, COLORS['lane_box'], rect, border_radius=8)
            pygame.draw.rect(screen, COLORS['white'], rect, 2, border_radius=8)

            signal_color = COLORS['green'] if current_signals[lane] == '🟢' else COLORS['red']
            pygame.draw.circle(screen, signal_color, (rect.centerx, rect.top - 30), 10)

            for i, _ in enumerate(vehicle_queues[lane]):
                car_width, car_height = 40, 20
                car_x = rect.centerx - car_width // 2
                car_y = rect.bottom - (car_height + 5) * (i + 1)
                pygame.draw.rect(
                    screen, COLORS['car'],
                    (car_x, car_y, car_width, car_height),
                    border_radius=4
                )

            label = font.render(f'{lane} | Cars: {len(vehicle_queues[lane])}', True, COLORS['white'])
            screen.blit(label, (rect.left + 10, rect.bottom + 10))

    def decide_next_lane():
        max_lane = max(LANES, key=lambda l: len(vehicle_queues[l]))
        now = time.time()
        candidates = []

        for lane in LANES:
            wait_time = now - last_green_time[lane]
            max_count = len(vehicle_queues[max_lane])
            lane_count = len(vehicle_queues[lane])
            if (lane_count > R_THRESHOLDS[lane] or
                    max_count - lane_count >= VEHICLE_DIFF_THRESHOLD or
                    wait_time > WAIT_LIMIT):
                candidates.append(lane)

        if candidates:
            return max(candidates, key=lambda l: len(vehicle_queues[l]))
        return max_lane

    def update_signals():
        nonlocal current_green_lane, green_start_time, current_signals

        now = time.time()
        if now - green_start_time < MIN_GREEN_TIME:
            return

        next_lane = decide_next_lane()
        if next_lane != current_green_lane:
            current_signals = {lane: '🔴' for lane in LANES}
            current_signals[next_lane] = '🟢'
            last_green_time[current_green_lane] = now
            current_green_lane = next_lane
            green_start_time = now

    def move_cars():
        if vehicle_queues[current_green_lane]:
            vehicle_queues[current_green_lane].pop(0)

    def add_random_vehicles():
        for lane in LANES:
            if random.random() < 0.1 and len(vehicle_queues[lane]) < MAX_CARS_PER_LANE:
                vehicle_queues[lane].append('🚗')

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        update_signals()

        if frame_count % MOVE_CARS_INTERVAL == 0:
            move_cars()

        if frame_count % ADD_VEHICLES_INTERVAL == 0:
            add_random_vehicles()

        draw_lanes()
        pygame.display.flip()
        clock.tick(30)
        frame_count += 1

    pygame.quit()


if __name__ == '__main__':
    main()
