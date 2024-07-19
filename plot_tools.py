import numpy as np
import matplotlib.pyplot as plt

def plot_points(*args, facecolors=None, alpha=None, capsize=10, elinewidth=10, capthick=6, dot_size=400, yticks=None, ytickslabels=None, ymin=None, ymax=None, random_state=10, figsize=(10,10)):
    np.random.seed(random_state)
    SPREAD_LENGTH = 0.1
    
    x_list = np.arange(1.0, 1.0+0.5*len(args), 0.5)
    y_list = list(map(lambda x: np.nanmean(x), args))
    error_list = list(map(lambda x: np.nanstd(x)/np.sqrt(len(x[~np.isnan(x)])), args))
    spread_list = x_list - SPREAD_LENGTH/2.0
    
    # create points for means and standard errors
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.errorbar(x_list, y_list, error_list, ls="none", capsize=capsize, elinewidth=elinewidth, capthick=capthick, ecolor="black")
    ax.scatter(x_list, y_list, s=dot_size, c="black", zorder=10)
    
    if facecolors is None:
        facecolors = np.repeat("#cccccc", len(args))
    if alpha is None:
        alpha = np.repeat(0.5, len(args))

    if type(facecolors) == str:
        facecolors = np.repeat(facecolors, len(args))
    if type(alpha) == float or type(alpha) == int:
        alpha = np.repeat(alpha, len(args))
    
    # draw each point
    for i, arg in enumerate(args):
        ax.scatter(
            np.random.rand(len(arg))*SPREAD_LENGTH + spread_list[i], 
            arg, 
            c=facecolors[i], 
            edgecolors='none', 
            s=dot_size, 
            alpha=alpha[i],
            zorder=0
        )

    ax.set_xlim([1.0-0.4, max(x_list)+0.4])
    ax.set_xticks(x_list, np.repeat("", len(args)))
    ax.spines[['right', 'top']].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=8)
    
    if ymin and ymax:
        ax.set_ylim([ymin, ymax])
    
    if yticks:
        if ytickslabels:
            ax.set_yticks(yticks, ytickslabels)
        else:
            ax.set_yticks(yticks)

def plot_correlation(x, y, lr, output_file=None, ymin=0, ymax=1, xmin=0, xmax=1, xticks=None, xtickslabels=None, yticks=None, ytickslabels=None, facecolor="blue", figsize=(6,6)):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.scatter(x, y, 400, c=facecolor, edgecolors="black", linewidth=3)
    ax.plot([xmin, xmax], [lr.slope*xmin+lr.intercept, lr.slope*xmax+lr.intercept], c="#333333", linewidth=5)
    ax.spines[['right', 'top']].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=6)
    
    if xmin and xmax:
        ax.set_xlim([xmin, xmax])
    
    if ymin and ymax:
        ax.set_ylim([ymin, ymax])
    
    if yticks and ytickslabels:
        ax.set_yticks(yticks, ytickslabels)
        
    if xticks and xtickslabels:
        ax.set_xticks(xticks, xtickslabels)
    
    if output_file:
        plt.tight_layout()
        plt.savefig(output_file)

def plot_confusion_matrix(m, output_path):
    plt.rcParams.update({'font.size': 60})

    labels_names = ["worse", "better"]
    fig, ax = plt.subplots(figsize=(12,12))
    im = ax.imshow(m, cmap="Blues")

    ax.spines[:].set_visible(False)
    ax.set_xticks([-0.5, 1.5])
    ax.set_yticks([-0.5, 1.5])

    ax.set_xticks([0.5],  minor=True)
    ax.set_yticks([0.5], minor=True)

    ax.set_xticklabels(["", ""])
    ax.set_yticklabels(["", ""])

    ax.grid(which="minor", color="w", linestyle='-', linewidth=15)

    for i in range(len(labels_names)):
        for j in range(len(labels_names)):
            text_color = "w" if i == j else "black"
            text = ax.text(j, i, round(m[i, j],2), ha="center", va="center", color=text_color)
            
    plt.tight_layout()
    plt.savefig(output_path)
