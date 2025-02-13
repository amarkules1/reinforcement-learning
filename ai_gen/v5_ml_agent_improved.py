import turtle
import time
import random
import numpy as np
import sys
import math

# --- Command line argument for eval mode ---
training_mode = True  # Default to training mode
if "--eval" in sys.argv:
    training_mode = False

# --- Set up screen (conditional UI) ---
render_ui = not training_mode  # Render UI only if not in training mode
if render_ui:
    screen = turtle.Screen()
    screen.title("Snake Game - ML Agent")
    screen.bgcolor("lightgreen")
    screen.setup(width=600, height=600)
    screen.tracer(0)

    # Score display
    score_display = turtle.Turtle()
    score_display.speed(0)
    score_display.color("black")
    score_display.penup()
    score_display.hideturtle()
    score_display.goto(0, 260)

# Snake head (always create, but only show if UI is rendered)
head = turtle.Turtle()
head.speed(0)
head.shape("square")
head.color("black")
head.penup()
head.goto(0, 0)
head.direction = "stop"
if not render_ui:
    head.hideturtle() # Hide head in training mode

# Snake food (always create, only show if UI is rendered)
food = turtle.Turtle()
food.speed(0)
food.shape("circle")
food.color("red")
food.penup()
food.shapesize(0.5, 0.5)
food.goto(0, 100)
if not render_ui:
    food.hideturtle() # Hide food in training mode

segments = []
obstacles = []

# Score
score = 0
high_score = 0

# Obstacle creation function
def create_obstacle():
    new_obstacle = turtle.Turtle()
    new_obstacle.speed(0)
    new_obstacle.shape("square")
    new_obstacle.color("darkgrey")
    new_obstacle.penup()
    new_obstacle.shapesize(1, 1)
    if not render_ui:
        new_obstacle.hideturtle() # Hide obstacles in training mode

    while True: # Ensure obstacle does not spawn on snake or food initially
        x = random.randint(-270, 270)
        y = random.randint(-270, 270)
        new_obstacle.goto(x, y)
        if head.distance(new_obstacle) > 20 and food.distance(new_obstacle) > 20:
            valid_position = True
            for seg in segments:
                if seg.distance(new_obstacle) < 20:
                    valid_position = False
                    break
            if valid_position:
                break
    obstacles.append(new_obstacle)

# Functions for snake movement
def go_up():
    if head.direction != "down":
        head.direction = "up"

def go_down():
    if head.direction != "up":
        head.direction = "down"

def go_left():
    if head.direction != "right":
        head.direction = "left"

def go_right():
    if head.direction != "left":
        head.direction = "right"

def move():
    if head.direction == "up":
        y = head.ycor()
        head.sety(y + 20)
    elif head.direction == "down":
        y = head.ycor()
        head.sety(y - 20)
    elif head.direction == "left":
        x = head.xcor()
        head.setx(x - 20)
    elif head.direction == "right":
        x = head.xcor()
        head.setx(x + 20)

# Keyboard bindings (for user control, can be disabled for agent - only in eval mode)
if render_ui:
    screen.listen()
    screen.onkeypress(go_up, "Up")
    screen.onkeypress(go_down, "Down")
    screen.onkeypress(go_left, "Left")
    screen.onkeypress(go_right, "Right")

# Game state variables
obstacle_timer = 0
obstacle_interval = 3
game_started = False
epsilon = 0.1 if training_mode else 0 # Exploration rate for training
learning_rate = 0.1
discount_factor = 0.9
q_table_file = "q_table.npy"
q_table = {} # Initialize Q-table as a dictionary
best_reward = -float('inf') # Track best reward during training
iteration_number = 0 # Track iteration number

# --- Q-Learning Agent ---
def get_state():
    """
    Returns a tuple representing the current game state.
    State is simplified for Q-Table:
    (food_direction_x, food_direction_y, danger_straight, danger_left, danger_right)
    food_direction_x: -1 (left), 0 (same x), 1 (right)
    food_direction_y: -1 (down), 0 (same y), 1 (up)
    danger_straight, danger_left, danger_right: 0 (no danger), 1 (danger)
    """
    head_x, head_y = head.xcor(), head.ycor()
    food_x, food_y = food.xcor(), food.ycor()

    food_direction_x = 0
    if food_x < head_x: food_direction_x = -1
    elif food_x > head_x: food_direction_x = 1

    food_direction_y = 0
    if food_y < head_y: food_direction_y = -1
    elif food_y > head_y: food_direction_y = 1

    # Danger detection (simplified - check immediate cell ahead, left, right)
    danger_straight = check_danger_ahead(head.direction)
    danger_left = check_danger_left(head.direction)
    danger_right = check_danger_right(head.direction)

    return (food_direction_x, food_direction_y, danger_straight, danger_left, danger_right)


def check_danger_ahead(direction):
    x, y = head.xcor(), head.ycor()
    if direction == "up":    y += 20
    elif direction == "down":  y -= 20
    elif direction == "left":  x -= 20
    elif direction == "right": x += 20
    return is_collision(x, y)

def check_danger_left(direction):
    x, y = head.xcor(), head.ycor()
    if direction == "up":    x -= 20
    elif direction == "down":  x += 20
    elif direction == "left":  y -= 20
    elif direction == "right": y += 20
    return is_collision(x, y)

def check_danger_right(direction):
    x, y = head.xcor(), head.ycor()
    x_forward, y_forward = get_forward_coords(head.direction) # Reuse forward coords for right check
    x, y = x_forward, y_forward
    if direction == "up":    x += 20
    elif direction == "down":  x -= 20
    elif direction == "left":  y += 20
    elif direction == "right": y -= 20
    return is_collision(x, y)

def get_forward_coords(direction):
    x, y = head.xcor(), head.ycor()
    if direction == "up":    y += 20
    elif direction == "down":  y -= 20
    elif direction == "left":  x -= 20
    elif direction == "right": x += 20
    return x, y


def is_collision(x, y):
    if x > 290 or x < -290 or y > 290 or y < -290: # Border collision
        return 1
    for segment in segments: # Self-collision
        if abs(x - segment.xcor()) < 20 and abs(y - segment.ycor()) < 20:
            return 1
    for obstacle in obstacles: # Obstacle collision
        if abs(x - obstacle.xcor()) < 20 and abs(y - obstacle.ycor()) < 20:
            return 1
    return 0 # No collision

def choose_action(state):
    if training_mode and random.uniform(0, 1) < epsilon:
        # Explore: choose a random action
        possible_actions = ["up", "down", "left", "right"]
        return random.choice(possible_actions)
    else:
        # Exploit: choose the best action from Q-table
        if state not in q_table:
            q_table[state] = {"up": 0, "down": 0, "left": 0, "right": 0} # Initialize if state is new
        return max(q_table[state], key=q_table[state].get)

def update_q_table(state, action, reward, next_state):
    if state not in q_table:
        q_table[state] = {"up": 0, "down": 0, "left": 0, "right": 0}
    if next_state not in q_table:
        q_table[next_state] = {"up": 0, "down": 0, "left": 0, "right": 0}

    best_next_q = max(q_table[next_state].values())
    q_table[state][action] = q_table[state][action] + learning_rate * (reward + discount_factor * best_next_q - q_table[state][action])

def save_q_table():
    np.save(q_table_file, q_table)
    print(f"Q-table saved to {q_table_file}")

def load_q_table():
    global q_table
    try:
        q_table = np.load(q_table_file, allow_pickle=True).item()
        print(f"Q-table loaded from {q_table_file}")
    except FileNotFoundError:
        print("Q-table file not found. Starting with an empty Q-table.")
        q_table = {}

# --- End Q-Learning Agent ---

def reset_game():
    global score, segments, obstacles, game_started, best_reward, iteration_number
    if render_ui:
        time.sleep(1)
    head.goto(0,0)
    head.direction = "stop"
    game_started = False

    for segment in segments:
        segment.goto(1000, 1000)
    segments.clear()
    for obstacle in obstacles:
        obstacle.goto(1000, 1000)
    obstacles.clear()

    iteration_number += 1 # Increment iteration count at each reset
    if score > best_reward: # Check for new best reward
        best_reward = score
        if training_mode:
            print(f"Iteration: {iteration_number}, New best reward: {best_reward}")
            save_q_table() # Save Q-table when new best reward is achieved

    score = 0
    if render_ui:
        score_display.clear()
        score_display.write(f"Score: {score}  High Score: {high_score}", align="center", font=("Courier", 24, "normal"))


# --- Improved Reward Function ---
def get_reward(current_state, action):
    reward = -0.01  # Small negative reward per step

    # Food proximity reward
    food_distance = head.distance(food)
    reward +=  (1 / food_distance) * 0.1 if food_distance > 0 else 0.1 # Higher reward for closer food

    # Obstacle distance reward (only if moving straight into potential danger)
    if action == "up" and current_state[0:3] == (0, 1, 1) or \
       action == "down" and current_state[0:3] == (0, -1, 1) or \
       action == "left" and current_state[0:3] == (-1, 0, 1) or \
       action == "right" and current_state[0:3] == (1, 0, 1): # Moving straight into danger
        x_forward, y_forward = get_forward_coords(head.direction)
        min_obstacle_distance_straight = float('inf')
        for obstacle in obstacles:
            distance_to_obstacle = math.sqrt((x_forward - obstacle.xcor())**2 + (y_forward - obstacle.ycor())**2)
            min_obstacle_distance_straight = min(min_obstacle_distance_straight, distance_to_obstacle)
        reward += (min_obstacle_distance_straight / 600) * 0.05 #  Reward proportional to distance from obstacle (scaled)


    if head.distance(food) < 15:
        reward = 1 + (1 / food_distance) * 0.1 # Big positive reward for food + food proximity bonus

    if head.xcor()>290 or head.xcor()<-290 or head.ycor()>290 or head.ycor()<-290: # Border
        reward = -1
    for segment in segments: # Self-collision
        if segment.distance(head) < 20:
            reward = -1
            break
    for obstacle in obstacles: # Obstacle collision
        if obstacle.distance(head) < 20:
            reward = -1
            break
    return reward
# --- End Improved Reward Function ---


# Main game loop
if training_mode:
    print("Training mode activated.")
    load_q_table() # Load existing Q-table to continue training
else:
    print("Evaluation mode activated. Agent will play using loaded knowledge.")
    load_q_table() # Load trained Q-table for evaluation
    epsilon = 0 # No exploration in eval mode


while True:
    if render_ui:
        screen.update()

    if game_started:
        obstacle_timer += 0.1
        if obstacle_timer >= obstacle_interval:
            create_obstacle()
            obstacle_timer = 0

        # Get current state
        current_state = get_state()

        # Agent chooses action
        action = choose_action(current_state)

        # Perform action
        if action == "up":    go_up()
        elif action == "down":  go_down()
        elif action == "left":  go_left()
        elif action == "right": go_right()

        move() # Move snake after action

        next_state = get_state() # State after action
        reward = get_reward(current_state, action) # Improved reward function

        # Check for game over conditions and food collision AFTER move
        game_over = False
        if head.xcor()>290 or head.xcor()<-290 or head.ycor()>290 or head.ycor()<-290: # Border
            game_over = True
        for segment in segments: # Self-collision
            if segment.distance(head) < 20:
                game_over = True
                break
        for obstacle in obstacles: # Obstacle collision
            if obstacle.distance(head) < 20:
                game_over = True
                break

        if head.distance(food) < 15: # Food collision
            x = random.randint(-270, 270)
            y = random.randint(-270, 270)
            food.goto(x, y)

            new_segment = turtle.Turtle()
            new_segment.speed(0)
            new_segment.shape("square")
            new_segment.color("grey")
            new_segment.penup()
            if not render_ui:
                new_segment.hideturtle()
            segments.append(new_segment)

            score += 10
            if score > high_score:
                high_score = score
            if render_ui:
                score_display.clear()
                score_display.write(f"Score: {score}  High Score: {high_score}", align="center", font=("Courier", 24, "normal"))


        if training_mode:
            update_q_table(current_state, action, reward, next_state)

        if game_over:
            if training_mode:
                update_q_table(current_state, action, reward, next_state) # Final update on game over
            reset_game()


        for index in range(len(segments)-1, 0, -1):
            x = segments[index-1].xcor()
            y = segments[index-1].ycor()
            segments[index].goto(x, y)
        if len(segments) > 0:
            x = head.xcor()
            y = head.ycor()
            segments[0].goto(x,y)


    else: # Game not started yet, agent needs to initiate movement
        if training_mode or epsilon == 0: # Agent starts automatically in training or eval
            game_started = True
            current_state = get_state()
            action = choose_action(current_state)
            if action == "up":    go_up()
            elif action == "down":  go_down()
            elif action == "left":  go_left()
            elif action == "right": go_right()

    if render_ui: # Control game speed based on mode
        time.sleep(0.1 if not training_mode else 0.02) # Original speed in eval, faster in training
    else:
        time.sleep(0.0001) # Minimal sleep in training without UI


# Save Q-table when closing if in training mode
if training_mode:
    save_q_table()
    print("Training complete. Q-table saved.")

if render_ui:
    screen.mainloop()