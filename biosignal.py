import pygame
import random
import time
import csv  


# Initialize
pygame.init()
screen = pygame.display.set_mode((800, 600)) #window size
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

#Center the row horizontally
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
        color = WHITE if i == highlight_idx else GRAY #If the current word’s index matches highlight_idx, make it WHITE (highlight).
        text = font.render(choice, True, color)
        rect = text.get_rect(center=positions[i]) #Calculates where to place the word
        screen.blit(text, rect)
    pygame.display.flip()
#paradigm start
# Ask user which one to focus on
print("Focus on one of the options: [0] Rock  [1] Paper  [2] Scissors")
target_idx = int(input("Enter the number of your target: "))

# Trial configuration
n_sequences = 15  # how many times to repeat full cycle
isi = 0.3         # inter-stimulus interval
flash_duration = 0.1

# Log to CSV
log_file = open("stimulus_log.csv", "w", newline='')
writer = csv.writer(log_file)
writer.writerow(["timestamp", "stimulus_idx", "stimulus_name", "is_target"])

# Run paradigm
for seq in range(n_sequences):
    order = [0, 1, 2]
    random.shuffle(order)
    for idx in order:
        draw_choice(highlight_idx=idx)
        timestamp = time.time()
        is_target = (idx == target_idx)
        writer.writerow([timestamp, idx, choices[idx], int(is_target)])
        time.sleep(flash_duration)
        draw_choice()
        time.sleep(isi)

log_file.close()
pygame.quit()