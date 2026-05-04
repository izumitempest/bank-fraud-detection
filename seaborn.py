import matplotlib.pyplot as plt
import numpy as np


def heatmap(data, annot=True, fmt=".3f", cmap="Blues", vmin=None, vmax=None, ax=None, cbar=False):
    if ax is None:
        ax = plt.gca()

    values = data.values if hasattr(data, "values") else np.asarray(data)
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    if hasattr(data, "columns"):
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_xticklabels(list(data.columns))
    if hasattr(data, "index"):
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_yticklabels(list(data.index))

    if annot:
        for row_index in range(values.shape[0]):
            for col_index in range(values.shape[1]):
                value = values[row_index, col_index]
                text_color = "white" if vmax is not None and value >= (vmax / 2) else "black"
                ax.text(col_index, row_index, format(value, fmt), ha="center", va="center", color=text_color, fontsize=9)

    if cbar:
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    return ax
