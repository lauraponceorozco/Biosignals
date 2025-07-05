# %%
# https://github.com/labstreaminglayer/liblsl-Python/blob/master/pylsl/pylsl.py
from pylsl import StreamInfo, StreamOutlet, local_clock

# %%
# Define the StreamInfo object
info = StreamInfo(
    name='test_outlet',
    type='eeg',
    channel_count=8,
    nominal_srate=250,
    channel_format='float32',
    source_id='test_id',
)

# Create the StreamOutlet object
outlet = StreamOutlet(info, chunk_size=0, max_buffered=360)

# Destroy the StreamInfo object to save space (optional)
info.__del__()

# %%
# Check whether there are consumers connected to the outlet
outlet.have_consumers()

# %%
# Wait for consumers to show up without wasting resources
# Returns True if the wait was successful, False if the timeout expired
# Only turns true if inlet.open_stream() was called
outlet.wait_for_consumers(timeout=10)

# %%
# Push one sample into the outlet
# Input: a list of values to push (one per channel)
tmp_list = []
for i in range(25000):
    before = local_clock()
    outlet.push_sample([1, 2, 3, 4, 5, 6, 7, 8])
    after = local_clock()
    tmp_list.append(after - before)
print(f'Clock before:\t{before}')
print(f'Clock after:\t{after}')
print(f'Elapsed time:\t{after-before}')

# %%
# Push a chunk of data
# Input: a list of lists or a list of multiplexed values
tmp_list = []
for i in range(2500):
    before = local_clock()
    outlet.push_chunk(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [11, 12, 13, 14, 15, 16, 17, 18],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [11, 12, 13, 14, 15, 16, 17, 18],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [11, 12, 13, 14, 15, 16, 17, 18],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [11, 12, 13, 14, 15, 16, 17, 18],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [11, 12, 13, 14, 15, 16, 17, 18],
        ]
    )
    after = local_clock()
    tmp_list.append(after - before)
print(f'Clock before:\t{before}')
print(f'Clock after:\t{after}')
print(f'Elapsed time:\t{after-before}')

# %%
# Destroy the outlet
# This is actually necessary to avoid a graveyard of dead outlets
outlet.__del__()
