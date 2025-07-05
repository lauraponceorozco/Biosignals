# %%
# https://github.com/labstreaminglayer/liblsl-Python/blob/master/pylsl/pylsl.py
from pylsl import StreamInlet, local_clock, resolve_byprop, resolve_streams

# %%
# Get the first StreamInfo object from a list of available streams
# info = resolve_streams()[0]

# Get the StreamInfo object by searching it by name
inlet_name = 'OpenViBE Stream EEG'
# inlet_name = 'test_outlet'

info = resolve_byprop('name', inlet_name)[0]

# Create a StreamInlet object
# max_buflen: Maximum amount of data to buffer in seconds
inlet = StreamInlet(info, max_buflen=60, max_chunklen=0)

# %%
# All the possible properties
print(f'Name: {inlet.info().name()}')
print(f'Type: {inlet.info().type()}')
print(f'Channel count: {inlet.info().channel_count()}')
print(f'Sampling rate: {inlet.info().nominal_srate()}')
print(f'Channel format: {inlet.info().channel_format()}')
print(f'Source ID: {inlet.info().source_id()}')
print(f'Protocol version: {inlet.info().version()}')
print(f'Stream created at: {inlet.info().created_at()}')
print(f'Unique ID of the outlet: {inlet.info().uid()}')
print(f'Session ID: {inlet.info().session_id()}')
print(f'Host name: {inlet.info().hostname()}')
print(f'Extended description: {inlet.info().desc()}')
# print(f'Stream info in XML format:\n{inlet.info().as_xml()}')

# %%
# Try to get the number of available samples
# Will be 0 because the inlet hasn't been opened yet
print(inlet.samples_available())

# Can be omitted, then pull_sample() or pull_chunk() calls it implicitly
# Let data flow in to the inlet
print('Opening stream at:', local_clock())
inlet.open_stream()

# %%
# Check the number of readily available data points
inlet.samples_available()

# %%
# Get a sample from the inlet
# Returns (None, None) if there was no new sample
# To remap the timestamp to the time stamp to the local clock, add the value
# returned by inlet.time_correction() to it.
sample, timestamp = inlet.pull_sample(timeout=0.00001)
print(f'Sample:\t{sample}')
print(f'Time stamp:\t{timestamp}')

# %%
# Get a chunk of samples from the inlet
# dest_obj can be a Python object that supports the buffer interface
samples, timestamps = inlet.pull_chunk(max_samples=5, dest_obj=None)
for i in zip(timestamps, samples):
    print(f'{i[0]}: {i[1]}\n')

# %%
# Check the estimated time correction offset of the stream
inlet.time_correction()

# %%
# Check the consistency of pull_chunk()
before = inlet.samples_available()
inlet.pull_chunk(max_samples=100)
after = inlet.samples_available()
print(f'# samples before pulling data:\t{before}')
print(f'# samples after pulling data:\t{after}')
print(f'# pulled samples:\t\t{before-after}')

# %%
# Close the inlet to further data points
# Stream cannot be reopened after this call
inlet.close_stream()

# Check the consistency of pull_chunk() after closing it to
# new samples
before = inlet.samples_available()
inlet.pull_chunk(max_samples=100)
after = inlet.samples_available()
print(f'# samples before pulling data:\t{before}')
print(f'# samples after pulling data:\t{after}')
print(f'# pulled samples:\t\t{before-after}')

# %%
# Drop all queued but not-yet pulled samples from the inlet
before = inlet.samples_available()
inlet.flush()
after = inlet.samples_available()
print(f'# samples before pulling data:\t{before}')
print(f'# samples after pulling data:\t{after}')

# %%
# Destroy the inlet (optional)
# The inlet will automatically disconnect if destroyed
inlet.__del__()
# %%
