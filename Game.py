import pygame
import random
import sys

"""
Rock-Paper-Scissors BCI Feedback GUI

Displays a result screen for a BCI classification output (0 = Rock, 1 = Paper, 2 = Scissors).
Shows both player and computer choices using hand emojis and highlights the outcome.
"""

# Setup
pygame.init()
screen = pygame.display.set_mode((800, 400))
pygame.display.set_caption("Rock Paper Scissors")
font_title = pygame.font.SysFont("Segoe UI Emoji", 60, bold=True)
font_text = pygame.font.SysFont("Segoe UI Emoji", 36)
font_result = pygame.font.SysFont("Arial", 64, bold=True)
clock = pygame.time.Clock()

# Options: Rock, Paper, Scissors
OPTIONS = ["✊", "✋", "✌️"]

def rps_result(player_choice):
    """
    Random computer choice, returns game result.
    """
    computer_choice = random.randint(0, 2)
    if player_choice == computer_choice:
        result = "Draw"
    elif (player_choice - computer_choice) % 3 == 1:
        result = "You Win!"
    else:
        result = "You Lose!"
    return player_choice, computer_choice, result

def draw_result(player_idx, comp_idx, result):
    """
    Displays player/computer choices and result.
    """
    screen.fill((245, 245, 245))

    # Title
    title = font_title.render("Rock Paper Scissors", True, (0, 0, 0))
    screen.blit(title, title.get_rect(center=(400, 60)))

    # Choices: Your vs My
    text_left = f"Your choice: {OPTIONS[player_idx]}"
    text_right = f"My choice: {OPTIONS[comp_idx]}"
    padding = 80

    render_left = font_text.render(text_left, True, (0, 0, 0))
    render_right = font_text.render(text_right, True, (0, 0, 0))

    total_width = render_left.get_width() + render_right.get_width() + padding
    start_x = (800 - total_width) // 2

    screen.blit(render_left, (start_x, 150))
    screen.blit(render_right, (start_x + render_left.get_width() + padding, 150))

    # Result color
    if result == "You Win!":
        result_color = (0, 160, 0)
    elif result == "You Lose!":
        result_color = (200, 0, 0)
    else:
        result_color = (0, 100, 200)

    # Result
    result_text = font_result.render(result, True, result_color)
    screen.blit(result_text, result_text.get_rect(center=(400, 270)))

    pygame.display.flip()

def main():
    """
    Runs one round of the GUI.
    """
    running = True
    simulated_input = random.randint(0, 2)  # Replace with classifier output

    player_idx, comp_idx, result = rps_result(simulated_input)
    draw_result(player_idx, comp_idx, result)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
