
import matplotlib.pyplot as plt
import numpy as np

def plot(data):

    # Setup plot
    plt.ion()

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    # Major ticks every 5, minor ticks every 1
    major_ticks = np.arange(0, 101, 5)
    minor_ticks = np.arange(0, 101, 1)

    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax1.grid(which='both')

    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.2)
    ax1.grid(which='major', alpha=0.5)

    line1, = ax1.plot(data)
    line2, = ax1.plot(data * 0.5)

    # plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
    # plt.pause(0.01)

    plt.show()

    return [plt, line1, line2]

# def show():
#     plt.show()