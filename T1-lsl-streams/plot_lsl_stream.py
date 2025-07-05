# %%
"""
ReceiveAndPlot example for LSL
This example shows data from all found outlets in realtime.
It illustrates the following use cases:
- efficiently pulling data, re-using buffers
- automatically discarding older samples
- online postprocessing
"""

import math
import sys
from typing import List

import numpy as np
import pylsl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# %%

# Basic parameters for the plotting window
plot_duration = 10  # how many seconds of data to show
update_interval = 30  # ms between screen updates
pull_interval = 20  # ms between each pull operation

# Define axis limits
ylims = None  # auto-range
# ylims = (-1, 1)  # fixed range

# Name of a single stream to plot (optional)
single_stream_name = ''
# single_stream_name = 'UN-2019.07.77_filtered'

# A specific string to search for in the stream names (optional)
stream_str = ''
# stream_str = 'UN'

# Choose to set the axis limits for a specific stream
lim_name = ''


class Inlet:
    """
    Base class to represent a plottable inlet
    """

    def __init__(self, info: pylsl.StreamInfo):
        # create an inlet and connect it to the outlet we found earlier.
        # max_buflen is set so data older the plot_duration is discarded
        # automatically and we only pull data new enough to show it

        # Also, perform online clock synchronization so all streams are in the
        # same time domain as the local lsl_clock()
        # (see https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html#_CPPv414proc_clocksync)
        # and dejitter timestamps
        self.inlet = pylsl.StreamInlet(
            info,
            max_buflen=plot_duration,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
        )
        # store the name and channel count
        self.name = info.name()
        self.channel_count = info.channel_count()
        self.channel_names = [
            'Channel ' + str(i + 1) for i in range(self.channel_count)
        ]
        # Only the first 8 channels are data channels in Unicorn
        if 'un' in self.name.lower() or self.channel_count == 17:
            self.channel_count = 8
            self.channel_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
        self.color_cycle = [
            '8dd3c7',
            'feffb3',
            'bfbbd9',
            'fa8174',
            '81b1d2',
            'fdb462',
            'b3de69',
            'bc82bd',
            'ccebc4',
            'ffed6f',
        ]

    def pull_and_plot(self, plot_time: float):
        """
        Pull data from the inlet and add it to the plot.
        :param plot_time: lowest timestamp that's still visible in the plot
        :param plt: the plot the data should be shown on
        """
        # We don't know what to do with a generic inlet, so we skip it.
        # It will be defined in the sub-classes
        pass


class DataInlet(Inlet):
    """
    A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted as multiple lines.
    """

    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo, win: pg.GraphicsLayoutWidget):
        super().__init__(info)

        self.win = win

        # calculate the size for our buffer, i.e. two times the displayed data
        bufsize = (
            2 * math.ceil(info.nominal_srate() * plot_duration),
            info.channel_count(),
        )
        self.buffer = np.empty(bufsize, dtype=self.dtypes[info.channel_format()])
        empty = np.array([])
        # create one curve object for each channel/line that will handle displaying the data
        self.curves = [
            pg.PlotCurveItem(
                x=empty,
                y=empty,
                #  autoDownsample=True,
                pen=pg.mkPen(self.color_cycle[i % len(self.color_cycle)], width=1),
            )
            for i in range(self.channel_count)
        ]

        offset = 0
        tmp = win.getItem(row=offset, col=0)
        while tmp is not None:
            offset = offset + 1
            tmp = self.win.getItem(row=offset, col=0)

        for i, curve in enumerate(self.curves):
            i_new = i + offset
            tmp_plt = self.win.addPlot(row=i_new, col=0, name=str(i))
            tmp_plt.setClipToView(True)
            tmp_plt.setDownsampling(mode='peak')
            tmp_plt.addItem(item=curve)
            tmp_plt.enableAutoRange(axis='x', enable=True)
            tmp_plt.setLabel('right', self.channel_names[i])
            tmp_plt.getAxis('right').setTicks([[]])
            if ylims is not None:
                if lim_name == '':
                    tmp_plt.setRange(yRange=ylims)
                else:
                    if lim_name.lower() in self.name.lower():
                        tmp_plt.setRange(yRange=ylims)
            if i_new > offset:
                tmp_plt.setXLink(str(i_new - 1))
            if i == 0:
                tmp_plt.setTitle(self.name)
            if i != len(self.curves) - 1:
                tmp_plt.getAxis('bottom').setTicks([[]])

    def pull_and_plot(self, plot_time):
        # pull the data
        _, ts = self.inlet.pull_chunk(
            timeout=0.0, max_samples=self.buffer.shape[0], dest_obj=self.buffer
        )
        # ts will be empty if no samples were pulled, a list of timestamps otherwise
        if ts:
            ts = np.asarray(ts)
            y = self.buffer[0 : ts.size, :]
            this_x = None
            old_offset = 0
            new_offset = 0
            for ch_ix in range(self.channel_count):
                # we don't pull an entire screen's worth of data, so we have to
                # trim the old data and append the new data to it
                old_x, old_y = self.curves[ch_ix].getData()
                # the timestamps are identical for all channels, so we need to # do this calculation only once
                if ch_ix == 0:
                    # find the index of the first sample that's still visible,
                    # i.e. newer than the left border of the plot
                    old_offset = old_x.searchsorted(plot_time)
                    # same for the new data, in case we pulled more data than
                    # can be shown at once
                    new_offset = ts.searchsorted(plot_time)
                    # append new timestamps to the trimmed old timestamps
                    this_x = np.hstack((old_x[old_offset:], ts[new_offset:]))
                # append new data to the trimmed old data
                this_y = np.hstack((old_y[old_offset:], y[new_offset:, ch_ix] - ch_ix))
                # replace the old data
                self.curves[ch_ix].setData(this_x, this_y)


class MarkerInlet(Inlet):
    """
    A MarkerInlet shows events that happen sporadically as vertical lines
    """

    def __init__(self, info: pylsl.StreamInfo, win: pg.GraphicsLayoutWidget):
        super().__init__(info)
        self.win = win

        i = 0
        tmp = win.getItem(row=i, col=0)
        while tmp is not None:
            i = i + 1
            tmp = self.win.getItem(row=i, col=0)

        self.plt = self.win.addPlot(row=i, col=0, name='markers')
        self.plt.setLabel('right', 'Markers')
        self.plt.getAxis('right').setTicks([[]])
        self.plt.getAxis('left').setTicks([[]])

        if i > 0:
            self.plt.setXLink(str(i - 1))
            self.win.getItem(row=i - 1, col=0).getAxis('bottom').setTicks([[]])

    def pull_and_plot(self, plot_time):
        # TODO: purge old markers
        strings, timestamps = self.inlet.pull_chunk(0)
        if timestamps:
            for string, ts in zip(strings, timestamps):
                self.plt.addItem(
                    pg.InfiniteLine(ts, angle=90, movable=False, label=string[0])
                )


def scroll():
    """
    Move the view so the data appears to scroll
    """
    # We show data only up to a timepoint shortly before the current time
    # so new data doesn't suddenly appear in the middle of the plot
    fudge_factor = pull_interval * 0.5
    plot_time = pylsl.local_clock()
    plt.setXRange(plot_time - plot_duration + fudge_factor, plot_time - fudge_factor)


def update():
    # Read data from the inlet. Use a timeout of 0.0 so we don't block GUI interaction.
    mintime = pylsl.local_clock() - plot_duration
    # call pull_and_plot for each inlet.
    # Special handling of inlet types (markers, continuous data) is done in
    # the different inlet classes.
    for inlet in inlets:
        inlet.pull_and_plot(mintime)


def get_stream_infos():
    print("Looking for streams...")

    # If there is a specific stream to be looked for
    if single_stream_name != '':
        # Specificially search for it
        infos = pylsl.resolve_byprop('name', single_stream_name)

    else:
        num_streams = 0

        # Stay in the while loop until streams are found
        while num_streams == 0:
            # Resolve all available LSL streams
            infos = pylsl.resolve_streams()
            num_streams = len(infos)

        # If there is a string to search for in the stream names
        if stream_str != '':
            # Filter out the streams with names without this string
            infos = [info for info in infos if stream_str in info.name()]

    # Sort the streams based on the stream type
    # --> marker streams at the end of the list
    infos.sort(key=lambda info: 'marker' in info.name().lower())

    return infos


def get_inlets(infos, win):
    # Initialize an empty list for the stream inlets
    inlets: List[Inlet] = []

    # Iterate over found streams, creating specialized inlet objects that will
    # handle plotting the data
    for info in infos:
        if 'marker' in info.type().lower():
            if (
                info.nominal_srate() != pylsl.IRREGULAR_RATE
                or info.channel_format() != pylsl.cf_string
            ):
                print('Invalid marker stream ' + info.name())
            print('Adding marker inlet: ' + info.name())
            inlets.append(MarkerInlet(info, win))
        elif (
            info.nominal_srate() != pylsl.IRREGULAR_RATE
            and info.channel_format() != pylsl.cf_string
        ):
            print('Adding data inlet: ' + info.name())
            inlets.append(DataInlet(info, win))
        else:
            print('Don\'t know what to do with stream ' + info.name())

    return inlets


if __name__ == '__main__':
    # Get a list of available LSL stream infos
    infos = get_stream_infos()

    # Create the pyqtgraph window
    win = pg.GraphicsLayoutWidget(show=True, title='LSL Data Stream')
    view = pg.widgets.RemoteGraphicsView.RemoteGraphicsView()
    pg.setConfigOptions(antialias=True)

    # Get a list of LSL inlets
    inlets = get_inlets(infos, win)

    # Get the first subplot for synchronizing with other subplots
    plt = win.getItem(row=0, col=0)

    # vb = win.addViewBox(row=0, col=0, colspan=inlets[0].channel_count)

    # Create a timer that will move the view every update_interval ms
    update_timer = QtCore.QTimer()
    update_timer.timeout.connect(scroll)
    update_timer.start(update_interval)

    # Create a timer that will pull and add new data occasionally
    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update)
    pull_timer.start(pull_interval)

    # Start Qt event loop unless running in interactive mode or using pyside
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()

# %%
