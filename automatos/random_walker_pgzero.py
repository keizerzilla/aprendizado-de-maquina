# random_walker_pgzero.py
# Artur Rodrigues
# 11/07/2021

import random

class RandomWalker:
    def __init__(self):
        self.x = int(WIDTH / 2)
        self.y = int(HEIGHT / 2)

    def walk(self):
        steps = [-1, 0, 1]

        self.x += random.choice(steps)
        self.y += random.choice(steps)

    def draw(self):
        box = Rect((self.x -3, self.y -3), (6, 6))
        color = (255, 255, 255)

        screen.draw.rect(box, color)

WIDTH = 480
HEIGHT = 320
walker = RandomWalker()

def update():
    walker.walk()

def draw():
    walker.draw()
