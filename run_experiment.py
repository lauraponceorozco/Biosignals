
from multiprocessing import Process, Event
import experiment_class as E


# initialization of variables
num_blocks = 5 # each block contains 3 trials (one for each target), every trial contains multiple epochs
num_epochs = 12 # each epoch contains all 3 stimuli in random order
flasing_duration = 0.1  
ISI = 0.5
# inter-stimulus interval is given by OpenVibe Designer!

csv_base_folder = "./data/exp3_ISI500/"  # base folder for CSV files


if __name__ == "__main__":

    exp = E.Experiment(num_blocks=num_blocks, 
                       num_epochs=num_epochs, 
                       flashing_duration=flasing_duration,
                       csv_base_folder=csv_base_folder,
                       inter_stimulus_interval=ISI)

    stop_event = Event()

    p1 = Process(target=exp.stream_eeg, args=(stop_event,))
    p2 = Process(target=exp.run_gui_local)

    p1.start()
    p2.start()

    # Wait for GUI to finish (stimulus list exhausted)
    p2.join()
    stop_event.set() # send stop signal to EEG stream process
    p1.join()

    print("All processes completed.")
