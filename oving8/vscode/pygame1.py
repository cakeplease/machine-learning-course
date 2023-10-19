# Example file showing a basic pygame "game loop"
import pygame
from gridWorld import Agent
# pygame setup
pygame.init()
x_agent = 25
y_agent = 225
WINDOW_WIDTH=400
WINDOW_HEIGHT = 300
MOVE_VALUE = 100
WHITE=(200,200,200)
YELLOW=(255,255,0)
ORANGE=(255,165,0)
RED=(200,0,0)
GREEN = (0, 0, 200)
BLACK=(0,0,0)
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
running = True
update = []

def drawNormalGrid():
    blockSize = MOVE_VALUE  # Set the size of the grid block
    for y in range(0, WINDOW_WIDTH, blockSize):
        for x in range(0, WINDOW_HEIGHT, blockSize):
            rect = pygame.Rect(y, x, blockSize, blockSize)
            pygame.draw.rect(screen, WHITE, rect, 1)


def drawGrid(state_values):
    blockSize = MOVE_VALUE  # Set the size of the grid block
    for y in range(0, WINDOW_WIDTH, blockSize):
        for x in range(0, WINDOW_HEIGHT, blockSize):
            for state in state_values:
                if (state == (round(x/100), round(y/100))):
                    if (state_values[state] >= -1 and state_values[state] < -0.5):
                        rect = pygame.Rect(y, x, blockSize, blockSize)
                        pygame.draw.rect(screen, WHITE, rect, 1)
                    elif (state_values[state] >= -0.5 and state_values[state] < 0):
                        rect = pygame.Rect(y, x, blockSize, blockSize)
                        pygame.draw.rect(screen, YELLOW, rect, 1)
                    elif (state_values[state] >= 0 and state_values[state] < 0.5):
                        rect = pygame.Rect(y, x, blockSize, blockSize)
                        pygame.draw.rect(screen, ORANGE, rect, 1)
                    elif (state_values[state] >= 0.5 and state_values[state] <= 1):
                        rect = pygame.Rect(y, x, blockSize, blockSize)
                        pygame.draw.rect(screen, RED, rect, 1)

print("SETUP")
drawNormalGrid()
# q learn setup
print("MODEL TRAIN")
model = Agent()
model.play()
print("MODEL RUN")

while model.State.isEnd == False:
    screen.fill("white")
    
    pygame.draw.rect(screen, GREEN, pygame.Rect(x_agent, y_agent, 50, 50), 1)
    pygame.draw.rect(screen, RED, pygame.Rect(325, 25, 50, 50), 1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = model.chooseAction()
    if (x_agent == WINDOW_WIDTH-MOVE_VALUE and y_agent == WINDOW_HEIGHT-MOVE_VALUE):
        model.State.isEnd = True
    else:
        print(action)
        if (x_agent < WINDOW_WIDTH and y_agent < WINDOW_HEIGHT and x_agent > 0 and y_agent > 0):
            if action == "right":
                x_agent += MOVE_VALUE
            if action == "left":
                x_agent -= MOVE_VALUE
            if action == "up":
                y_agent -= MOVE_VALUE
            if action == "down":
                y_agent += MOVE_VALUE
        if (x_agent > WINDOW_WIDTH and y_agent < WINDOW_HEIGHT and y_agent > 0):
            x_agent -= MOVE_VALUE
        if (x_agent < WINDOW_WIDTH and y_agent > WINDOW_HEIGHT and x_agent > 0):
            y_agent -= MOVE_VALUE
        if (y_agent > WINDOW_HEIGHT and x_agent > WINDOW_WIDTH):
            x_agent -= MOVE_VALUE
            y_agent -= MOVE_VALUE
        if (x_agent < 0 and y_agent < WINDOW_HEIGHT and y_agent > 0):
            x_agent += MOVE_VALUE
        if (x_agent < WINDOW_WIDTH and y_agent < 0 and x_agent > 0):
            y_agent += MOVE_VALUE
        if (x_agent < 0 and y_agent<0):
            x_agent += MOVE_VALUE
            y_agent += MOVE_VALUE

        model.State = model.takeAction(action)
        model.State.isEndFunc()
   
    # flip() the display to put your work on screen
    drawGrid(model.state_values)
    pygame.display.flip()

    clock.tick(1)  # limits FPS to 60

print("FINISH")
pygame.quit()
