import types
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pandas.api.types import is_numeric_dtype
import Constants

def plot_mesures_table(dfi):
    if(not Constants.VERBOSE_PLOT_ENABLED): return
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    dfi.insert(0,"Feature",dfi.index)
    table = ax.table(cellText = dfi.values, colLabels = dfi.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for i in range(len(dfi.columns)): table[(0, i)].set_facecolor("#56b5fd")
    for i in range(len(dfi.index) + 1): table[(i, 0)].set_facecolor("#56b5fd")
    fig.tight_layout()
    plt.show()

def plot_features_charts(df,rows, cols):
    if(not Constants.VERBOSE_PLOT_ENABLED): return
    grid = GridSpec(rows, cols, left=0.1, bottom=0.1, right=0.94, top=0.94, wspace=0.3, hspace=1)
    fig = plt.figure(0)
    fig.clf()
    plots = [[0 for x in range(cols)] for y in range(rows)] 
    
    for row in range(rows):
        for col in range(cols):
            plots[row][col] = fig.add_subplot(grid[row, col])
            plots[row][col].tick_params(axis = "x", labelrotation = 50)

    feature_idx = 0
    for row in range(rows):
        for col in range(cols):
            if(feature_idx == len(df.columns)): break
            plots[row][col].title.set_text(df.columns[feature_idx])
            if(is_numeric_dtype(df.iloc[:,feature_idx])):
                plots[row][col].hist(df.iloc[:,feature_idx], facecolor = "r")
            else:
                plots[row][col].bar(df.iloc[:,feature_idx].value_counts().keys().tolist(),
                                    df.iloc[:,feature_idx].value_counts().tolist() , facecolor = "b")   
            feature_idx+=1

    plt.show()

def plot_MI_scores(scores):
    if(not Constants.VERBOSE_PLOT_ENABLED): return
    x, y = scores["Feature"], scores["MI_Score"]
    plt.barh(x, y, facecolor = "g")
    plt.title("Features mutual information score")
    plt.xlabel("MI Scores")
    plt.ylabel("Features")
    plt.show()

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText = scores.values, colLabels = scores.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for i in range(len(scores.columns)): table[(0, i)].set_facecolor("#56b5fd")
    for i in range(len(scores.index) + 1): table[(i, 0)].set_facecolor("#56b5fd")
    fig.tight_layout()
    plt.show()

def plot_tsne_scatter(X_embedded, y, dim):
    if(not Constants.VERBOSE_PLOT_ENABLED): return
    labels = ["Reliable", "Unreliable"]
    if(dim == Constants.TSNE_2D):
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = y)
    elif(dim == Constants.TSNE_3D):
        plt.axes(projection = "3d")
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1],X_embedded[:, 2], c = y)
    handles, _ = scatter.legend_elements(prop = "colors")
    plt.legend(handles, labels)
    plt.show()

def plot_scaled_plot(df):
    if(not Constants.VERBOSE_PLOT_ENABLED): return
    df.plot(kind = 'density')
    plt.show()

def plot_models_scores(dfi):
    if(not Constants.VERBOSE_PLOT_ENABLED): return
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText = dfi.values, colLabels = dfi.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for i in range(len(dfi.columns)): table[(0, i)].set_facecolor("#56b5fd")
    for i in range(len(dfi.index) + 1): table[(i, 0)].set_facecolor("#56b5fd")
    fig.tight_layout()
    plt.show()

def plot_fitting_chart(ts, vs, xlbl, ylbl, title):
    if(not Constants.VERBOSE_PLOT_ENABLED): return
    plt.plot(ts.keys(), ts.values(), color = "b", label = "Training-set")
    plt.plot(vs.keys(), vs.values(), color = "r", label = "Validation-set")
    plt.scatter(ts.keys(), ts.values(), s = 30, facecolors = "none", edgecolors = "b")
    plt.scatter(vs.keys(), vs.values(), s = 30, facecolors = "none", edgecolors = "r")
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_confusion_matrix(conf_matrix):
    if(not Constants.VERBOSE_PLOT_ENABLED): return
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap = plt.cm.Blues, alpha = 0.8)
    ax.xaxis.set_label_position('top') 
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predicted class', fontsize = 18)
    plt.ylabel('Real class', fontsize = 18)
    plt.show()