import joblib
import pygame

from coms import *
from snake_ents import Snake, Food


class SnakeGame:
    def __init__(self):
        self.snake = Snake()
        self.food = Food()
        self.x_size = 500
        self.y_size = 500
        self.x_lim = self.x_size - 10
        self.y_lim = self.y_size - 10
        self.screen = pygame.display.set_mode((self.x_size, self.y_size))
        self.clock = pygame.time.Clock()
        self.points = 0
        self.level = 0
        self.bg = (0, 0, 0)
        self.turns = 0
        self.end_flag = False
        self.radio = 10
        self.verbose = False

    def check_events(self):
        if self.snake.head_position[0] < 0 or self.snake.head_position[0] > self.x_lim or self.snake.head_position[
            1] < 0 or \
                self.snake.head_position[1] > self.y_lim or (self.snake.head_position in self.snake.body):
            self.points = self.points - 5
            self.end_flag = True
        if self.snake.head_position == self.food.position:
            self.food.go_new_position()
            self.snake.grow()
            self.turns = 0
            self.points += 3
        if self.points >= 1000 * 3:
            self.end_flag = True

    def screen_draw(self):
        self.screen.fill(self.bg)
        x0 = int(self.snake.head_position[0]) - self.radio * 10
        x1 = int(self.snake.head_position[0]) + self.radio * 10
        y0 = int(self.snake.head_position[1]) - self.radio * 10
        y1 = int(self.snake.head_position[1]) + self.radio * 10
        pygame.draw.lines(self.screen, self.snake.vision_raidio_color, True, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        pygame.draw.rect(self.screen, self.snake.head_color,
                         (self.snake.head_position[0], self.snake.head_position[1], 10, 10))
        for b in self.snake.body:
            pygame.draw.rect(self.screen, self.snake.body_color, (b[0], b[1], 10, 10))
        pygame.draw.rect(self.screen, self.food.color, (self.food.position[0], self.food.position[1], 10, 10))

    def check_user_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.end_flag = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and self.snake.direction != 2:
                    self.snake.direction = 3
                if event.key == pygame.K_UP and self.snake.direction != 1:
                    self.snake.direction = 0
                if event.key == pygame.K_DOWN and self.snake.direction != 0:
                    self.snake.direction = 1
                if event.key == pygame.K_LEFT and self.snake.direction != 3:
                    self.snake.direction = 2

    def ai_directive(self, direction):
        if direction == 3 and self.snake.direction != 2:
            self.snake.direction = direction
        if direction == 0 and self.snake.direction != 1:
            self.snake.direction = direction
        if direction == 1 and self.snake.direction != 0:
            self.snake.direction = direction
        if direction == 2 and self.snake.direction != 3:
            self.snake.direction = direction

    def play(self):
        """
        Playable version of snake

        """
        pygame.display.set_caption("Super snake bros 3 The Fall of Reach: Revenge")
        self.food.go_new_position()
        while True:
            pygame.display.flip()
            self.clock.tick(80)
            self.check_user_events()
            self.snake.move()
            self.screen_draw()
            self.check_events()
            if self.end_flag:
                return self.points

    def board2set(self):
        xdir = 0
        if self.food.position[0] > self.snake.head_position[0]:
            xdir = 1
        elif self.food.position[0] < self.snake.head_position[0]:
            xdir = -1
        ydir = 0
        if self.food.position[1] > self.snake.head_position[1]:
            ydir = 1
        if self.food.position[1] < self.snake.head_position[1]:
            ydir = -1
        uc = [self.snake.head_position[0], self.snake.head_position[1] - 10]
        dc = [self.snake.head_position[0], self.snake.head_position[1] + 10]
        lc = [self.snake.head_position[0] - 10, self.snake.head_position[1]]
        rc = [self.snake.head_position[0] + 10, self.snake.head_position[1]]
        cords = [uc, dc, lc, rc]
        sens = [0, 0, 0, 0]
        for i, cord in enumerate(cords):
            if cord[1] < 0 or cord[1] > self.y_lim or cord[0] < 0 or cord[0] > self.x_lim:
                sens[i] = 4
            if cord == self.food.position:
                sens[i] = 3
            if cord in self.snake.body:
                sens[i] = 2
        s = sens + [self.snake.direction, xdir, ydir]
        return s

    def demo(self, settler):
        pygame.display.set_caption("Super snake bros 3 The Fall of Reach: Revenge")
        self.food.go_new_position()
        while True:
            pygame.display.flip()
            old_board = self.board2set()
            old_points = self.points
            old_dist = dist(self.snake.head_position, self.food.position)
            policy = settler.interact(old_board)
            self.ai_directive(policy)
            self.snake.move()
            self.check_events()
            self.clock.tick(80)
            self.screen_draw()
            settler.update_q(
                old_board,
                self.board2set(),
                state2int(self.points - old_points,
                          dp_reward(old_dist, dist(self.snake.head_position, self.food.position))),
                policy
            )
            if self.end_flag:
                return settler

    def demo_light(self, settler):
        self.food.go_new_position()
        while True:
            old_board = self.board2set()
            old_points = self.points
            old_dist = dist(self.snake.head_position, self.food.position)
            policy = settler.interact(old_board)
            self.ai_directive(policy)
            self.snake.move()
            self.check_events()
            settler.update_q(
                old_board,
                self.board2set(),
                state2int(self.points - old_points,
                          dp_reward(old_dist, dist(self.snake.head_position, self.food.position))),
                policy
            )
            if self.end_flag:
                return settler


if __name__ == '__main__':
    game = SnakeGame()
    ai = joblib.load("memory.pkl")
    ai.learning_rate = 0
    ai.gamma = 0
    ai.epsilon = 0
    game.demo(ai)
    # game.play()
