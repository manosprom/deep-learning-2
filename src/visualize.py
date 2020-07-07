def plot_history(histories, metrics=["loss", "auc", "accuracy", "kappa"], figsize=(20, 10)):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10})

    rows = int(np.ceil(len(metrics) / 2))
    columns = 2
    fig, ax = plt.subplots(rows, columns, figsize=figsize)

    for i, metric in enumerate(metrics):
        row = int(i / columns)
        column = int(i % columns)
        for label in histories:
            ax[row, column].plot(histories[label][metric], label='{0:s} train {1:s}'.format(label, metric), linewidth=2)
            ax[row, column].plot(histories[label]['val_{0:s}'.format(metric)], label='{0:s} validation {1:s}'.format(label, metric), linewidth=2)
            ax[row, column].set_xticks(np.arange(0, len(histories[label][metric]), 2))
        ax[row, column].legend()
    plt.close(fig)
    return fig
