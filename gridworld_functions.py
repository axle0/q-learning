# these libraries are required
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

# these libraries are useful to use
import copy
import math


# this function can be called to display the Q values for each state and action available at each state
# it will draw an image showing the Q values at each state and return this image
# you are not required to alter this function however are welcome to if you wish
def draw_qs(qs):
    # error checking the q array passed in
    try:
        if qs.shape != (5, 5, 4):
            raise ValueError()
    except:
        print('Q table has incorrect shape')
        exit(0)

    # initialise the drawing
    q_grid = Image.new(mode='RGBA', size=(1001, 1001), color=0)
    draw = ImageDraw.Draw(q_grid)
    draw.rectangle(((0, 0), (1000, 1000)), fill='black', outline='green')

    y1 = 0
    y2 = 1000
    x1 = 0
    for i in range(5):
        x2 = x1
        draw.line(((x1, y1), (x2, y2)))
        x1 += 200

    x1 = 0
    x2 = 1000
    y1 = 0
    for i in range(5):
        y2 = y1
        draw.line(((x1, y1), (x2, y2)))
        y1 += 200

    grids_coord = []
    x1 = 0
    x2 = 200
    y1 = 0
    y2 = 200
    for i in range(5):
        grid_one = []
        grid_one.append((x1, y1))
        grid_one.append((x2, y2))
        grids_coord.append(grid_one)
        y1_temp = y1
        y2_temp = y2
        for j in range(4):
            y1_temp = y2_temp
            y2_temp += 200
            grid_one_ = []
            grid_one_.append((x1, y1_temp))
            grid_one_.append((x2, y2_temp))
            grids_coord.append(grid_one_)
        x1 = x2
        x2 += 200

    for grid_boundary in grids_coord:
        draw.line(((grid_boundary[0][0], grid_boundary[0][1]), (grid_boundary[1][0], grid_boundary[1][1])))
        draw.line(((grid_boundary[1][0], grid_boundary[0][1]), (grid_boundary[0][0], grid_boundary[1][1])))

    x = 0
    y = 0
    fnt = None
    try:
        fnt = ImageFont.truetype("arial.ttf", 20)
    except:
        fnt = ImageFont.truetype("FreeMono.ttf", 28, encoding="unic")
    for grid in grids_coord:
        # print(grid)

        # print(qs[x][y][:])

        q_left = str(qs[x][y][0])

        x1 = grid[0][0] + 25
        y1 = (grid[0][1] + grid[1][1]) / 2
        draw.text((x1, y1), q_left, font=fnt)

        q_up = str(qs[x][y][1])
        x1 = grid[0][0] + 90
        y1 = grid[0][1] + 20
        draw.text((x1, y1), q_up, font=fnt)

        q_right = str(qs[x][y][2])
        x1 = grid[0][0] + 140
        y1 = (grid[0][1] + grid[1][1]) / 2
        draw.text((x1, y1), q_right, font=fnt)

        q_down = str(qs[x][y][3])
        x1 = grid[0][0] + 90
        y1 = grid[0][1] + 140
        draw.text((x1, y1), q_down, font=fnt)

        x += 1
        if x == 5:
            x = 0
            y += 1

    # filling terminal states
    draw.rectangle(((200, 200), (400, 400)), fill='red')
    draw.rectangle(((600, 200), (800, 400)), fill='red')
    draw.rectangle(((400, 600), (600, 800)), fill='red')
    draw.rectangle(((0, 600), (200, 800)), fill='red')
    draw.rectangle(((800, 800), (1000, 1000)), fill='lightblue')

    # defining font for terminals states
    fnt2 = ImageFont.truetype("arial.ttf", 30)

    # drawing terminal state rewards
    draw.text((875, 880), '+1.0', font=fnt2, fill='black')
    draw.text((75, 680), '-1.0', font=fnt2, fill='black')
    draw.text((475, 680), '-1.0', font=fnt2, fill='black')
    draw.text((275, 280), '-1.0', font=fnt2, fill='black')
    draw.text((675, 280), '-1.0', font=fnt2, fill='black')

    return q_grid


# this function creates an environment being a 5 by 5 gridworld, and places mines at particular locations in the environment
# it returns the environment ( a numpy array), a list that contains tuples which contain the positions of the relevant mines, and a tuple containing the goal location
def create_environment():
    env = np.chararray(shape=(5, 5), unicode=True)
    mines = 0
    mine_positions = []
    # mine_positions.append((1,0))
    mine_positions.append((1, 1))
    mine_positions.append((1, 3))
    mine_positions.append((3, 0))
    mine_positions.append((3, 2))
    for i in range(5):
        for j in range(5):
            ismine = 0
            for a in mine_positions:
                if i == a[0] and j == a[1]:
                    env[i][j] = 'X'
                    ismine = 1
            if ismine == 0:
                env[i][j] = '-'
    goal = (4, 4)
    env[goal[0]][goal[1]] = 'G'
    env[0][0] = 'A'
    return env, mine_positions, goal


# this function will take the agents current position,
# its chosen action, its environment, and the locations of the mines/goal in the environment
# it will update the environment and return the new position of the agent
# the action scheme works as follows
# 0 = left, 1 = up, 2 = right, 3 = down
def render_agent(position, action, env, mines, goal):
    pos = copy.deepcopy(position)
    old_pos = copy.deepcopy(pos)
    # 0 left, 1 up, 2 right, 3 down
    if action == 0:
        if pos[1] == 0:
            # dont alter agent x coordinate as agent is at left hand boundary of gridworld
            pass
        else:
            pos[1] -= 1

    elif action == 1:
        if pos[0] == 0:
            ##dont alter agent y coordinate as agent is at top boundary of gridworld
            pass
        else:
            pos[0] -= 1

    elif action == 2:
        if pos[1] == 4:
            pass
            # done alter agent x coordinate as agent is at right boundary of gridworld
        else:
            pos[1] += 1

    elif action == 3:
        if pos[0] == 4:
            # done alter agent x coordinate as agent is at bottom boundary of gridworld
            pass
        else:
            pos[0] += 1

    # check if agent has hit a mine
    for m_ in mines:
        mine_x = m_[0]
        mine_y = m_[1]
        if pos[0] == mine_x and pos[1] == mine_y:
            # hit mine
            print('hit mine')
            # you will need to specify an appropriate penalty under the variable name reward
            reward = -1

            # finish the episode
            done = True

            # return new position, reward, and done flag
            return pos, reward, done

    # check if agent has hit the goal state
    if pos[0] == goal[0] and pos[1] == goal[1]:
        print('goal')
        # you will need to specify an appropriate reward
        reward = 1
        done = True
        return pos, reward, done


    # the agent has not hit a terminal state, continue episode
    else:
        # specify the reward received
        reward = -0.1
        done = False
        # delete old agent position in env
        for s in range(5):
            for u in range(5):
                if env[s][u] == 'A':
                    env[s][u] = '-'
        # udate agent position in env
        env[pos[0]][pos[1]] = 'A'
        return pos, reward, done


def raiseNotDefined(function_name):
    print('error, function with name ', function_name, ' has not been implemented')


def draw_policy(qs):
    # error checking the q array passed in
    try:
        if qs.shape != (5, 5):
            raise ValueError()
    except:
        print('Policy table has incorrect shape')
        exit(0)

    # initialise the drawing
    q_grid = Image.new(mode='RGBA', size=(1001, 1001), color=0)
    draw = ImageDraw.Draw(q_grid)
    draw.rectangle(((0, 0), (1000, 1000)), fill='black', outline='green')

    y1 = 0
    y2 = 1000
    x1 = 0
    for i in range(5):
        x2 = x1
        draw.line(((x1, y1), (x2, y2)))
        x1 += 200

    x1 = 0
    x2 = 1000
    y1 = 0
    for i in range(5):
        y2 = y1
        draw.line(((x1, y1), (x2, y2)))
        y1 += 200

    grids_coord = []
    x1 = 0
    x2 = 200
    y1 = 0
    y2 = 200
    for i in range(5):
        grid_one = []
        grid_one.append((x1, y1))
        grid_one.append((x2, y2))
        grids_coord.append(grid_one)
        y1_temp = y1
        y2_temp = y2
        for j in range(4):
            y1_temp = y2_temp
            y2_temp += 200
            grid_one_ = []
            grid_one_.append((x1, y1_temp))
            grid_one_.append((x2, y2_temp))
            grids_coord.append(grid_one_)
        x1 = x2
        x2 += 200

    x = 0
    y = 0
    fnt = None
    try:
        fnt = ImageFont.truetype("arial.ttf", 20)
    except:
        fnt = ImageFont.truetype("FreeMono.ttf", 28, encoding="unic")

    for grid in grids_coord:

        q = str(qs[x][y])
        if q == 'r':
            x1 = grid[0][0] + 80
            y1 = (grid[0][1] + grid[1][1]) / 2
            draw.polygon(
                [(x1, y1), (x1, y1 + 10), (x1 + 30, y1 + 10), (x1 + 30, y1 + 20), (x1 + 50, y1), (x1 + 30, y1 - 20),
                 (x1 + 30, y1 - 10), (x1, y1 - 10)], fill='blue')
        elif q == 'l':
            x1 = grid[0][0] + 120
            y1 = (grid[0][1] + grid[1][1]) / 2
            draw.polygon(
                [(x1, y1), (x1, y1 + 10), (x1 - 30, y1 + 10), (x1 - 30, y1 + 20), (x1 - 50, y1), (x1 - 30, y1 - 20),
                 (x1 - 30, y1 - 10), (x1, y1 - 10)], fill='blue')

        elif q == 'd':
            x1 = grid[0][0] + 100
            y1 = (grid[0][1] + grid[1][1]) / 2 - 20
            draw.polygon(
                [(x1, y1), (x1 - 10, y1), (x1 - 10, y1 + 30), (x1 - 20, y1 + 30), (x1, y1 + 50), (x1 + 20, y1 + 30),
                 (x1 + 10, y1 + 30), (x1 + 10, y1)], fill='blue')

        elif q == 'u':
            x1 = grid[0][0] + 100
            y1 = (grid[0][1] + grid[1][1]) / 2 + 20
            draw.polygon(
                [(x1, y1), (x1 - 10, y1), (x1 - 10, y1 - 30), (x1 - 20, y1 - 30), (x1, y1 - 50), (x1 + 20, y1 - 30),
                 (x1 + 10, y1 - 30), (x1 + 10, y1)], fill='blue')

        elif q == 'R':
            x1 = grid[0][0] + 80
            y1 = (grid[0][1] + grid[1][1]) / 2
            draw.polygon(
                [(x1, y1), (x1, y1 + 10), (x1 + 30, y1 + 10), (x1 + 30, y1 + 20), (x1 + 50, y1), (x1 + 30, y1 - 20),
                 (x1 + 30, y1 - 10), (x1, y1 - 10)], fill='red')
        elif q == 'L':
            x1 = grid[0][0] + 120
            y1 = (grid[0][1] + grid[1][1]) / 2
            draw.polygon(
                [(x1, y1), (x1, y1 + 10), (x1 - 30, y1 + 10), (x1 - 30, y1 + 20), (x1 - 50, y1), (x1 - 30, y1 - 20),
                 (x1 - 30, y1 - 10), (x1, y1 - 10)], fill='red')

        elif q == 'D':
            x1 = grid[0][0] + 100
            y1 = (grid[0][1] + grid[1][1]) / 2 - 20
            draw.polygon(
                [(x1, y1), (x1 - 10, y1), (x1 - 10, y1 + 30), (x1 - 20, y1 + 30), (x1, y1 + 50), (x1 + 20, y1 + 30),
                 (x1 + 10, y1 + 30), (x1 + 10, y1)], fill='red')

        elif q == 'U':
            x1 = grid[0][0] + 100
            y1 = (grid[0][1] + grid[1][1]) / 2 + 20
            draw.polygon(
                [(x1, y1), (x1 - 10, y1), (x1 - 10, y1 - 30), (x1 - 20, y1 - 30), (x1, y1 - 50), (x1 + 20, y1 - 30),
                 (x1 + 10, y1 - 30), (x1 + 10, y1)], fill='red')
        x += 1
        if x == 5:
            x = 0
            y += 1

    # filling terminal states
    draw.rectangle(((200, 200), (400, 400)), fill='red')
    draw.rectangle(((600, 200), (800, 400)), fill='red')
    draw.rectangle(((400, 600), (600, 800)), fill='red')
    draw.rectangle(((0, 600), (200, 800)), fill='red')
    draw.rectangle(((800, 800), (1000, 1000)), fill='lightblue')

    # defining font for terminals states
    fnt2 = ImageFont.truetype("arial.ttf", 30)

    # drawing terminal state rewards
    draw.text((875, 880), '+1.0', font=fnt2, fill='black')
    draw.text((75, 680), '-1.0', font=fnt2, fill='black')
    draw.text((475, 680), '-1.0', font=fnt2, fill='black')
    draw.text((275, 280), '-1.0', font=fnt2, fill='black')
    draw.text((675, 280), '-1.0', font=fnt2, fill='black')

    return q_grid
