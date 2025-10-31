import random
import numpy as np

class SnakeGame:

    def __init__(self, grid=10, ai=False):
        self.ai = ai
        self.reset(grid)
        self.direction = [1,0]  # initial direction to the right
        self.reward = 0
        if self.ai: self.starv = grid**2
        else: self.starv = 10000
        self.hunger = 0
        self.game_over = False
        self.vision = 0
        self.segSize = 3

    def get_reward(self, event):
        match event:
            case "apple":
                return 1
            case "self":
                return -1
            case "wall":
                return -1
            case "starv":
                return -.5
            case _:
                return -0.001 * (self.hunger+1)

    def reset(self, grid):
        self.grid_size = grid
        self.snake_position =[{"x": random.randint(0, self.grid_size-1), "y": random.randint(0, self.grid_size-1)}]
        self.food_position = self.setFoodPosition()
        self.score = 0
        self.hunger = 0
        return self.score, self.food_position, self.snake_position

    def setFoodPosition(self):
        while True:
            # use strings "x" and "y" as keys
            position = {"x": random.randint(0, self.grid_size - 1),
                        "y": random.randint(0, self.grid_size - 1)}
            # check if any segment of the snake already uses that position
            if all(segment["x"] != position["x"] or segment["y"] != position["y"]
                for segment in self.snake_position):
                return position

    def get_food_pos(self):
        food_x = self.food_position["x"]/self.grid_size
        food_y = self.food_position["y"]/self.grid_size
        return food_x, food_y
    
    def get_head_pos(self):
        head_x = self.snake_position[0]["x"]/self.grid_size
        head_y = self.snake_position[0]["y"]/self.grid_size
        return head_x, head_y

    def get_danger(self, sight=0):
        head = self.snake_position[0]
        snake_set = {(s["x"], s["y"]) for s in self.snake_position[1:]}

        if sight == 0:
            # [up, down, left, right] if the **next** cell that way is dangerous
            danger = [0, 0, 0, 0]

            # walls
            if head["y"] == 0:                   danger[0] = 1  # up would go OOB
            if head["y"] == self.grid_size - 1:  danger[1] = 1  # down OOB
            if head["x"] == 0:                   danger[2] = 1  # left OOB
            if head["x"] == self.grid_size - 1:  danger[3] = 1  # right OOB

            # body (immediate neighbors only)
            if (head["x"], head["y"] - 1) in snake_set: danger[0] = 1
            if (head["x"], head["y"] + 1) in snake_set: danger[1] = 1
            if (head["x"] - 1, head["y"]) in snake_set: danger[2] = 1
            if (head["x"] + 1, head["y"]) in snake_set: danger[3] = 1

            return danger
        if sight > 0:
            # sight > 0: return a flat list for the window (2*sight+1)^2 around head
            danger = []
            for dy in range(-sight, sight + 1):
                for dx in range(-sight, sight + 1):
                    nx = head["x"] + dx
                    ny = head["y"] + dy

                    # if you want the center cell to always be safe:
                    if dx == 0 and dy == 0:
                        continue

                    out_of_bounds = (nx < 0 or ny < 0 or nx >= self.grid_size or ny >= self.grid_size)
                    hits_body     = (nx, ny) in snake_set
                    danger.append(1 if out_of_bounds or hits_body else 0)
            return danger
        else:
            danger = []
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if head["x"] == x and head["y"] == y: 
                        danger.append(-1)
                    elif (x, y) in snake_set:
                        danger.append(-0.5)
                    elif self.food_position["x"] == x and self.food_position["y"] == y: 
                        danger.append(1)
                    else:
                        danger.append(0)
            return danger
        
    def get_segment(self, seg_size = 0):
        seg_x = self.snake_position[0]["x"]
        seg_y = self.snake_position[0]["y"]
        segs = []
        seg_length = (len(self.snake_position)-1)//seg_size
        for i in range(1,seg_size):
            dif_x = (self.snake_position[i*seg_length]["x"] - seg_x)
            dif_y = (self.snake_position[i*seg_length]["y"] - seg_y)
            seg_x = self.snake_position[i*seg_length]["x"]
            seg_y = self.snake_position[i*seg_length]["y"]
            segs.append(dif_x)
            segs.append(dif_y)
        dif_x = (self.snake_position[len(self.snake_position)-1]["x"] - seg_x)
        dif_y = (self.snake_position[len(self.snake_position)-1]["y"] - seg_y)
        segs.append(dif_x)
        segs.append(dif_y)
        return segs
            


    def step(self, action):
        # Current head (copy so we don't mutate in-place before checks)
        head = self.snake_position[0].copy()

        # Move intent
        if action == 'up':
            head["y"] -= 1
            self.direction = [0,-1]
        elif action == 'down':
            head["y"] += 1
            self.direction = [0,1]
        elif action == 'left':
            head["x"] -= 1
            self.direction = [-1,0]
        elif action == 'right':
            head["x"] += 1
            self.direction = [1,0]
        elif action == 0:  # straight
            head["x"] += self.direction[0]
            head["y"] += self.direction[1]
        elif action == 1:  # right turn
            if self.direction == [1,0]:   # right -> down
                head["x"] += 0
                head["y"] += 1
                self.direction = [0,1]
            elif self.direction == [0,1]: # down -> left
                head["x"] -= 1
                head["y"] += 0
                self.direction = [-1,0]
            elif self.direction == [-1,0]:# left -> up
                head["x"] += 0
                head["y"] -= 1
                self.direction = [0,-1]
            elif self.direction == [0,-1]:# up -> right
                head["x"] += 1
                head["y"] += 0
                self.direction = [1,0]
        elif action == 2:  # left turn
            if self.direction == [1,0]:   # right -> up
                head["x"] += 0
                head["y"] -= 1
                self.direction = [0,-1]
            elif self.direction == [0,1]: # down -> right
                head["x"] += 1
                head["y"] += 0
                self.direction = [1,0]
            elif self.direction == [-1,0]:# left -> down
                head["x"] += 0
                head["y"] += 1
                self.direction = [0,1]
            elif self.direction == [0,-1]:# up -> left
                head["x"] -= 1
                head["y"] += 0
                self.direction = [-1,0]

        # Wall collision
        if (head["x"] < 0 or head["x"] >= self.grid_size or
            head["y"] < 0 or head["y"] >= self.grid_size):
            self.game_over = True
            self.reward = self.get_reward("wall")
            return self.score, self.food_position, self.snake_position, self.game_over

        # Self collision (compare against existing segments)
        if any(seg["x"] == head["x"] and seg["y"] == head["y"]
            for seg in self.snake_position):
            self.game_over = True
            self.reward = self.get_reward("self")
            return self.score, self.food_position, self.snake_position, self.game_over

        # Apply movement: insert new head
        self.snake_position.insert(0, head)

        # Food check
        if head["x"] == self.food_position["x"] and head["y"] == self.food_position["y"]:
            self.score += 1
            self.reward = self.get_reward("apple")
            self.hunger = 0
            self.food_position = self.setFoodPosition()  # grow (don't pop tail)
        else:
            self.reward = self.get_reward("")
            self.hunger += 1
            if self.hunger >= self.starv:
                self.game_over = True
                self.reward = self.get_reward("starv")
                return self.score, self.food_position, self.snake_position, self.game_over
            self.snake_position.pop()  # no food: move by dropping tail

        return self.score, self.food_position, self.snake_position, self.game_over
    
    def get_state(self):
        apple_channel = []
        head_channel = []
        body_channel = []
        channel = []
        for i in range(self.grid_size):
            apple_channel.append([])
            head_channel.append([])
            body_channel.append([])
            channel.append([])
            for j in range(self.grid_size):
                apple_channel[i].append(0)
                head_channel[i].append(0)
                body_channel[i].append(0)
                channel[i].append(0)
                
        #apple_
        channel[self.food_position["x"]][self.food_position["y"]] = 1
        #head_
        channel[self.snake_position[0]["x"]][self.snake_position[0]["y"]] = -1
        for i in self.snake_position[1:]:
            #body_
            channel[i["x"]][i["y"]] = -.5

        apple_x = (self.food_position["x"] - self.snake_position[0]["x"])/self.grid_size
        apple_y = (self.food_position["y"] - self.snake_position[0]["y"])/self.grid_size
        length = len(self.snake_position)-1
        tail_x = (self.snake_position[length]["x"] - self.snake_position[0]["x"])/self.grid_size
        tail_y = (self.snake_position[length]["x"] - self.snake_position[0]["x"])/self.grid_size
        mid_x = (self.snake_position[int(length/2)]["x"] - self.snake_position[0]["x"])/self.grid_size
        mid_y = (self.snake_position[int(length/2)]["x"] - self.snake_position[0]["x"])/self.grid_size
        danger = self.get_danger(sight=self.vision)
        segs = self.get_segment(seg_size=self.segSize)
        size_x = (max(s["x"] for s in self.snake_position) - min(s["x"] for s in self.snake_position))/self.grid_size
        size_y = (max(s["y"] for s in self.snake_position) - min(s["y"] for s in self.snake_position))/self.grid_size
        size = len(self.snake_position)/(self.grid_size**2)
        board = []
        for y in range(self.grid_size):
            board.append([])

        state = [
            apple_x,
            apple_y,
            #tail_x,
            #tail_y,
            self.direction[0],
            self.direction[1],
            self.hunger/self.starv,
            #size
        ]
        for i in segs:
            state.append(i/self.grid_size)

        for i in danger:
            state.append(i)
            #for z in range(self.grid_size):
                #board[z].append(i)
        return state, self.reward, self.game_over, [board]