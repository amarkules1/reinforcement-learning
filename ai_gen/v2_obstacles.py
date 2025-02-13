import turtle
import time
import random

# Set up screen
screen = turtle.Screen()
screen.title("Snake Game Harder")
screen.bgcolor("lightgreen")
screen.setup(width=600, height=600)
screen.tracer(0)  # Turns off screen updates

# Snake head
head = turtle.Turtle()
head.speed(0)
head.shape("square")
head.color("black")
head.penup()
head.goto(0, 0)
head.direction = "stop"

# Snake food
food = turtle.Turtle()
food.speed(0)
food.shape("circle")
food.color("red")
food.penup()
food.shapesize(0.5, 0.5)
food.goto(0, 100)

segments = []
obstacles = []

# Score
score = 0
high_score = 0

score_display = turtle.Turtle()
score_display.speed(0)
score_display.color("black")
score_display.penup()
score_display.hideturtle()
score_display.goto(0, 260)
score_display.write(f"Score: {score}  High Score: {high_score}", align="center", font=("Courier", 24, "normal"))

# Obstacle creation function
def create_obstacle():
    new_obstacle = turtle.Turtle()
    new_obstacle.speed(0)
    new_obstacle.shape("square")
    new_obstacle.color("darkgrey")
    new_obstacle.penup()
    new_obstacle.shapesize(1, 1)

    while True: # Ensure obstacle does not spawn on snake or food initially (not perfect but better)
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

    if head.direction == "down":
        y = head.ycor()
        head.sety(y - 20)

    if head.direction == "left":
        x = head.xcor()
        head.setx(x - 20)

    if head.direction == "right":
        x = head.xcor()
        head.setx(x + 20)

# Keyboard bindings
screen.listen()
screen.onkeypress(go_up, "Up")
screen.onkeypress(go_down, "Down")
screen.onkeypress(go_left, "Left")
screen.onkeypress(go_right, "Right")

# Game state variables
obstacle_timer = 0
obstacle_interval = 3  # seconds

# Main game loop
while True:
    screen.update()

    # Check for obstacle creation time
    obstacle_timer += 0.1 # assuming time.sleep(0.1) below
    if obstacle_timer >= obstacle_interval:
        create_obstacle()
        obstacle_timer = 0

    # Check for collision with border
    if head.xcor()>290 or head.xcor()<-290 or head.ycor()>290 or head.ycor()<-290:
        time.sleep(1)
        head.goto(0,0)
        head.direction = "stop"

        # Hide segments and obstacles
        for segment in segments:
            segment.goto(1000, 1000)
        segments.clear()
        for obstacle in obstacles:
            obstacle.goto(1000, 1000) # move off screen
        obstacles.clear()


        # Reset score
        score = 0
        score_display.clear()
        score_display.write(f"Score: {score}  High Score: {high_score}", align="center", font=("Courier", 24, "normal"))


    # Check for collision with food
    if head.distance(food) < 15:
        # Move the food to a random spot
        x = random.randint(-270, 270)
        y = random.randint(-270, 270)
        food.goto(x, y)

        # Add a segment
        new_segment = turtle.Turtle()
        new_segment.speed(0)
        new_segment.shape("square")
        new_segment.color("grey")
        new_segment.penup()
        segments.append(new_segment)

        # Increase the score
        score += 10

        if score > high_score:
            high_score = score

        score_display.clear()
        score_display.write(f"Score: {score}  High Score: {high_score}", align="center", font=("Courier", 24, "normal"))


    # Move the end segments first in reverse order
    for index in range(len(segments)-1, 0, -1):
        x = segments[index-1].xcor()
        y = segments[index-1].ycor()
        segments[index].goto(x, y)

    # Move segment 0 to where the head is
    if len(segments) > 0:
        x = head.xcor()
        y = head.ycor()
        segments[0].goto(x,y)

    move()

    # Check for head collision with body segments
    for segment in segments:
        if segment.distance(head) < 20:
            time.sleep(1)
            head.goto(0,0)
            head.direction = "stop"

             # Hide segments and obstacles
            for segment in segments:
                segment.goto(1000, 1000)
            segments.clear()
            for obstacle in obstacles:
                obstacle.goto(1000, 1000) # move off screen
            obstacles.clear()

            # Reset score
            score = 0
            score_display.clear()
            score_display.write(f"Score: {score}  High Score: {high_score}", align="center", font=("Courier", 24, "normal"))
            break # Break out of segment collision loop after reset

    # Check for head collision with obstacles
    for obstacle in obstacles:
        if head.distance(obstacle) < 20:
            time.sleep(1)
            head.goto(0,0)
            head.direction = "stop"

            # Hide segments and obstacles
            for segment in segments:
                segment.goto(1000, 1000)
            segments.clear()
            for obstacle in obstacles:
                obstacle.goto(1000, 1000) # move off screen
            obstacles.clear()

            # Reset score
            score = 0
            score_display.clear()
            score_display.write(f"Score: {score}  High Score: {high_score}", align="center", font=("Courier", 24, "normal"))
            break # Break out of obstacle collision loop after reset


    time.sleep(0.1) # Delay to control game speed

screen.mainloop()