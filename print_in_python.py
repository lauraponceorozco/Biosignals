from pylsl import resolve_byprop
streams = resolve_byprop('type', 'EEG', timeout=5)
for s in streams:
    print("Stream found:", s.name(), s.type(), s.source_id())

from pylsl import resolve_byprop, StreamInlet
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import time

# Parameters
BUFFER_SIZE = 500  # Number of samples to display
CHANNEL_NAMES = ['F3', 'O1', 'O2']  # Adjust based on your OpenViBE stream
NUM_CHANNELS = len(CHANNEL_NAMES)

# Resolve EEG stream
streams = resolve_byprop('type', 'EEG', timeout=5)
if not streams:
    raise RuntimeError("No EEG stream found.")
print("Stream found:", streams[0].name())

# Create an inlet to read from the stream
inlet = StreamInlet(streams[0])

# Initialize buffers
data_buffers = [deque([0]*BUFFER_SIZE, maxlen=BUFFER_SIZE) for _ in range(NUM_CHANNELS)]

# Setup plot
plt.ion()
fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, figsize=(10, 6))
lines = []
for i in range(NUM_CHANNELS):
    line, = ax[i].plot(range(BUFFER_SIZE), data_buffers[i])
    ax[i].set_ylabel(CHANNEL_NAMES[i])
    lines.append(line)
ax[-1].set_xlabel("Samples")

# Live plot loop
while True:
    sample, timestamp = inlet.pull_sample(timeout=1.0)
    if sample:
        for i in range(NUM_CHANNELS):
            data_buffers[i].append(sample[i])
            lines[i].set_ydata(data_buffers[i])
        plt.pause(0.001)