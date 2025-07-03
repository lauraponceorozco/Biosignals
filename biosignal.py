import pygame
import random
import time
import csv  


pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("P300 Rock Paper Scissors")

# Colors and fonts
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
BLACK = (0, 0, 0)
font = pygame.font.SysFont(None, 80)

choices = ['Rock', 'Paper', 'Scissors']

# Measure text sizes and compute total width
text_surfaces = [font.render(choice, True, GRAY) for choice in choices]
text_widths = [surf.get_width() for surf in text_surfaces]
spacing = 60
total_width = sum(text_widths) + spacing * (len(choices) - 1)

start_x = (screen.get_width() - total_width) // 2
y_pos = screen.get_height() // 2

# Compute positions dynamically
positions = []
x = start_x
for width in text_widths:
    positions.append((x + width // 2, y_pos))
    x += width + spacing


# Drawing Function

def draw_choice(highlight_idx=None):
    screen.fill(BLACK)
    for i, choice in enumerate(choices):
        color = WHITE if i == highlight_idx else GRAY
        text = font.render(choice, True, color)
        rect = text.get_rect(center=positions[i])
        screen.blit(text, rect)
    pygame.display.flip() #updates display


# GUI-Based Target Selection

def select_target():
    screen.fill(BLACK)
    prompt_font = pygame.font.SysFont(None, 40)
    instructions = prompt_font.render("Press 1 for Rock, 2 for Paper, 3 for Scissors", True, WHITE)
    screen.blit(instructions, instructions.get_rect(center=(400, 100)))

    for i, choice in enumerate(choices):
        text = font.render(choice, True, GRAY)
        rect = text.get_rect(center=positions[i])
        screen.blit(text, rect)

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return 0  # Rock
                elif event.key == pygame.K_2:
                    return 1  # Paper
                elif event.key == pygame.K_3:
                    return 2  # Scissors


# Main Experiment Loop


# Step 1: Ask user for focus target
target_idx = select_target()

# Step 2: Trial settings
n_sequences = 15  # number of full random sequences
flash_duration = 0.1
isi = 0.3  # inter-stimulus interval

# Step 3: Prepare log
log_file = open("stimulus_log.csv", "w", newline='')
writer = csv.writer(log_file)
writer.writerow(["timestamp", "stimulus_idx", "stimulus_name", "is_target"])

# Step 4: Run paradigm
for seq in range(n_sequences):
    order = [0, 1, 2]
    random.shuffle(order)
    for idx in order:
        draw_choice(highlight_idx=idx) #draws the words, highlights one word 
        timestamp = time.time() #records time the word was highlighted
        is_target = int(idx == target_idx) #logs if it matches the users chosen target
        writer.writerow([timestamp, idx, choices[idx], is_target])
        time.sleep(flash_duration)
        draw_choice()  # no highlight
        time.sleep(isi)

log_file.close()
pygame.quit()
