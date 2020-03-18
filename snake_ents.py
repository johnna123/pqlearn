from random import randint


class Snake:
    def __init__(self):
        self.head_position = [0, 0]
        self.body = []
        self.direction = 3
        self.grow_position = [-20, -20]
        self.body_color = (42, 65, 255)
        self.head_color = (107, 26, 210)
        self.vision_raidio_color = (96, 96, 96)

    def move(self):
        if len(self.body) > 0:
            self.body.append(self.head_position.copy())
            self.body.remove(self.body[0])
        if self.direction == 3:
            self.head_position[0] += 10
        elif self.direction == 2:
            self.head_position[0] -= 10
        elif self.direction == 0:
            self.head_position[1] -= 10
        elif self.direction == 1:
            self.head_position[1] += 10

    def grow(self):
        self.body.insert(0, self.grow_position)


class Food:
    def __init__(self):
        self.position = [100, 100]
        self.color = (255, 42, 65)

    def go_new_position(self):
        self.position = [randint(0, 49) * 10, randint(0, 49) * 10]
