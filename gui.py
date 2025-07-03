import pygame
import random
import time

# Initialize
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Rock Paper Scissors Flash")

# Colors and font
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
BLACK = (0, 0, 0)
font = pygame.font.SysFont(None, 80)

choices = ['Rock', 'Paper', 'Scissors']

# Measure text sizes and compute total width
text_surfaces = [font.render(choice, True, GRAY) for choice in choices]
text_widths = [surf.get_width() for surf in text_surfaces]
spacing = 60  # space between words
total_width = sum(text_widths) + spacing * (len(choices) - 1)

# Compute starting x so it’s centered
start_x = (screen.get_width() - total_width) // 2
y_pos = screen.get_height() // 2

# Compute positions dynamically
positions = []
x = start_x
for width in text_widths:
    positions.append((x + width // 2, y_pos))
    x += width + spacing

def draw_choice(highlight_idx=None):
    screen.fill(BLACK)
    for i, choice in enumerate(choices):
        color = WHITE if i == highlight_idx else GRAY
        text = font.render(choice, True, color)
        rect = text.get_rect(center=positions[i])
        screen.blit(text, rect)
    pygame.display.flip()

# Main loop: flash each choice randomly
for _ in range(20):
    idx = random.randint(0, 2)
    draw_choice(highlight_idx=idx)
    time.sleep(0.1)
    draw_choice()  # no highlight
    time.sleep(0.4)

time.sleep(1)
pygame.quit()
