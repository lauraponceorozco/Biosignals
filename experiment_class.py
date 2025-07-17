from pylsl import StreamInlet, resolve_byprop
import time
import pygame
import random
import csv
from datetime import datetime
import os


class Experiment:

    def __init__(self, num_blocks=3, num_epochs=12, inter_stimulus_interval=0.3, flashing_duration=0.1,
                 csv_base_folder="./data/test/"):
        self.num_blocks = num_blocks
        self.num_epochs = num_epochs
        self.inter_stimulus_interval = inter_stimulus_interval
        self.flashing_duration = flashing_duration

        # create a timestamped folder for CSV files (to avoid overwriting)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        csv_folder = os.path.join(csv_base_folder, f"session_{timestamp}")
        os.makedirs(csv_folder, exist_ok=False) #create folder
        self.eeg_csv_path = os.path.join(csv_folder, "eeg_data.csv")
        self.gui_csv_path = os.path.join(csv_folder, "gui_data.csv")

    def shuffle_stimuli(self, num_batches):
        # returns a list of stimuli: each batch contains the stimuli [0, 1, 2] in random order -> 3x num_batches stimuli

        base_stimuli = [0, 1, 2]
        stimulus_list = []

        for _ in range(num_batches):
            epoch_stimuli = base_stimuli[:]
            random.shuffle(epoch_stimuli)
            stimulus_list.extend(epoch_stimuli)

        return stimulus_list


    def stream_eeg(self, stop_event=None):
        print("Looking for an EEG stream...")

        # Resolve EEG stream
        eeg_streams = resolve_byprop("name", "openvibeSignal", timeout=5)
        print(eeg_streams)
        if not eeg_streams:
            print("No EEG stream found.")
            return

        eeg_inlet = StreamInlet(eeg_streams[0])
        samples = []

        try:
            while not (stop_event and stop_event.is_set()):
                eeg_sample, eeg_ts = eeg_inlet.pull_sample(timeout=0.0)
                if eeg_sample:
                    time_ux = time.time()
                    samples.append([time_ux, eeg_ts] + eeg_sample)
                    


        finally:
            output_path = self.eeg_csv_path
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp_ux', 'timestamp'] + [f'channel_{i}' for i in range(len(samples[0]) - 2)])
                writer.writerows(samples)
            print(f"Saved {len(samples)} samples to {output_path}")

           
    def run_gui(self):
        

        # create a target list for the GUI
        target_list = self.shuffle_stimuli(self.num_blocks)

        # Initialize
        pygame.init()
        screen = pygame.display.set_mode((1280, 800))
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
        spacing = 200  # space between words
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

        
        def draw_choice_text(highlight_idx=None):
            screen.fill(BLACK)
            for i, choice in enumerate(choices):
                color = WHITE if i == highlight_idx else GRAY
                text = font.render(choice, True, color)
                rect = text.get_rect(center=positions[i])
                screen.blit(text, rect)
            pygame.display.flip()
        
        def draw_choice_box(highlight_idx=None):
            screen.fill(BLACK)
            for i, choice in enumerate(choices):
                is_highlight = (i == highlight_idx)
                text_color = WHITE if is_highlight else GRAY
                text = font.render(choice, True, text_color)
                rect = text.get_rect(center=positions[i])

                if is_highlight:
                    padding = 20
                    box_rect = pygame.Rect(
                        rect.left - padding,
                        rect.top - padding,
                        rect.width + 2 * padding,
                        rect.height + 2 * padding
                    )
                    pygame.draw.rect(screen, WHITE, box_rect)  # filled box
                    screen.blit(text, rect)  # text over box (same color)
                else:
                    screen.blit(text, rect)

            pygame.display.flip()

        # Resolve marker stream
        marker_streams = resolve_byprop("name", "openvibeMarkers", timeout=5)
        print(marker_streams)
        marker_inlet = StreamInlet(marker_streams[0])

        stim_rec = []
        try:
            
            for trial, target_idx in enumerate(target_list):

                # Display loop until key press
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            waiting = False

                    screen.fill(BLACK)

                    # Draw "Look at ..." text
                    instruction_text = font.render("Look at ...", True, WHITE)
                    instruction_rect = instruction_text.get_rect(center=(screen.get_width() // 2, 100))
                    screen.blit(instruction_text, instruction_rect)

                    # Draw choices with target highlighted in red
                    for i, choice in enumerate(choices):
                        color = (255, 0, 0) if i == target_idx else GRAY
                        text = font.render(choice, True, color)
                        rect = text.get_rect(center=positions[i])
                        screen.blit(text, rect)

                    pygame.display.flip()
                    pygame.time.wait(50)

                    #create a stimulus list for the current target
                    stimulus_list = self.shuffle_stimuli(self.num_epochs)

                # Flush any previously buffered markers
                while marker_inlet.pull_sample(timeout=0.0)[0] is not None:
                    continue

                while True:
                    marker_sample, marker_ts = marker_inlet.pull_sample(timeout=0.0)
                    if marker_sample is not None:
                        if stimulus_list:
                            print(f"Marker: {marker_ts}, {marker_sample[0]}")
                            idx = stimulus_list.pop(0)
                            time_ux = time.time()
                            draw_choice_box(highlight_idx=idx)# choose between flasing text of boxes using _box or _text
                            time.sleep(self.flashing_duration)
                            draw_choice_box()  # no highlight

                            # bool value indicates if stim corresponds to target
                            is_target = int(idx == target_idx) #(redundant, but for clarity)

                            stim_rec.append([trial, time_ux, marker_ts, target_idx, idx, is_target]) #only record marker when stimulus is displayed, only time and idx of interest
                        else:
                            print("All stimuli displayed, exiting GUI.")
                            break

        finally:
                output_path = self.gui_csv_path
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['trial', 'time_ux','timestamp', 'target', 'stimulus', 'is_target'])
                    writer.writerows(stim_rec)
                print(f"Saved {len(stim_rec)} marker-stimulus pairs to {output_path}")
                    
    def run_gui_local(self):
        

        # create a target list for the GUI
        target_list = self.shuffle_stimuli(self.num_blocks)

        # Initialize
        pygame.init()
        screen = pygame.display.set_mode((1280, 800))
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
        spacing = 200  # space between words
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

        
        def draw_choice_text(highlight_idx=None):
            screen.fill(BLACK)
            for i, choice in enumerate(choices):
                color = WHITE if i == highlight_idx else GRAY
                text = font.render(choice, True, color)
                rect = text.get_rect(center=positions[i])
                screen.blit(text, rect)
            pygame.display.flip()
        
        def draw_choice_box(highlight_idx=None):
            screen.fill(BLACK)
            for i, choice in enumerate(choices):
                is_highlight = (i == highlight_idx)
                text_color = WHITE if is_highlight else GRAY
                text = font.render(choice, True, text_color)
                rect = text.get_rect(center=positions[i])

                if is_highlight:
                    padding = 20
                    box_rect = pygame.Rect(
                        rect.left - padding,
                        rect.top - padding,
                        rect.width + 2 * padding,
                        rect.height + 2 * padding
                    )
                    pygame.draw.rect(screen, WHITE, box_rect)  # filled box
                    screen.blit(text, rect)  # text over box (same color)
                else:
                    screen.blit(text, rect)

            pygame.display.flip()

        stim_rec = []
        try:
            
            for trial, target_idx in enumerate(target_list):

                # Display loop until key press
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            waiting = False

                    screen.fill(BLACK)

                    # Draw "Look at ..." text
                    instruction_text = font.render("Look at ...", True, WHITE)
                    instruction_rect = instruction_text.get_rect(center=(screen.get_width() // 2, 100))
                    screen.blit(instruction_text, instruction_rect)

                    # Draw choices with target highlighted in red
                    for i, choice in enumerate(choices):
                        color = (255, 0, 0) if i == target_idx else GRAY
                        text = font.render(choice, True, color)
                        rect = text.get_rect(center=positions[i])
                        screen.blit(text, rect)

                    pygame.display.flip()
                    pygame.time.wait(50)

                    #create a stimulus list for the current target
                    stimulus_list = self.shuffle_stimuli(self.num_epochs)

                last_time = time.time()
                while True:
                    current_time = time.time()
                    if current_time - last_time >= self.inter_stimulus_interval:
                        if stimulus_list:
                            idx = stimulus_list.pop(0)
                            draw_choice_text(highlight_idx=idx)# choose between flasing text of boxes using _box or _text
                            time.sleep(self.flashing_duration)
                            draw_choice_text()  # no highlight

                            # bool value indicates if stim corresponds to target
                            is_target = int(idx == target_idx) #(redundant, but for clarity)

                            stim_rec.append([trial, current_time, target_idx, idx, is_target]) #only record marker when stimulus is displayed, only time and idx of interest
                        else:
                            print("All stimuli displayed, exiting GUI.")
                            break

        finally:
                output_path = self.gui_csv_path
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['trial','timestamp', 'target', 'stimulus', 'is_target'])
                    writer.writerows(stim_rec)
                print(f"Saved {len(stim_rec)} marker-stimulus pairs to {output_path}")
                    
