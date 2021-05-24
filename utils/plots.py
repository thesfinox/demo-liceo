import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(context='notebook', palette='tab10')

def subplots(nrows=1, ncols=1, **kwargs):
    '''
    Create plot grid.
    
    Arguments:
    
        nrows:  the no. of rows,
        ncols:  the no. of columns,
        kwargs: additional arguments for plt.subplots.
    
    Returns:
        
        figure, axes.
    '''
    
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows), **kwargs)

def plot_dimensionality_reduced(data,
                                x,
                                y,
                                labs=None,
                                style=None,
                                xlab='x',
                                ylab='y',
                                title='',
                                pal='tab10',
                                ax=None
                               ):
    '''
    Print the dimensionality reduced plot.
    
    Arguments:
    
        data:  the DataFrame,
        x:     the column on the x axis,
        y:     the column on the y axis,
        labs:  (the cluster labels, the number of labels),
        style: the style labels,
        xlab:  the labels of the x axis,
        ylab:  the label of the y axis,
        title: the title of the plot,
        pal:   palette,
        ax:    the axis.
    '''
    
    if ax is None:
        
        _, ax = subplots()

    sns.scatterplot(data=data,
                    x=x,
                    y=y,
                    hue=labs[0],
                    style=style,
                    palette=sns.color_palette(pal, n_colors=labs[1]),
                    ax=ax
                   )

    ax.set(xlabel=xlab, ylabel=ylab, title=title)
    ax.legend(bbox_to_anchor=(1.0, 0.0), loc='lower left')
    
    plt.tight_layout()
    
def plot_precision_recall(prec, rec, ax=None):
    '''
    Plot precision vs recall.
    
    Arguments:
    
        prec: precision,
        rec:  recallm
        ax:   the axis.
    '''
    
    if ax is None:
        
        _, ax = subplots()
    
    sns.lineplot(x=rec,
                 y=prec,
                 ci=None,
                 ax=ax
                )

    ax.set(xlabel='recall', ylabel='precision', title='Precision vs Recall')
    
    plt.tight_layout()
    
def plot_roc_curve(fpr, tpr, score, ax=None):
    '''
    Plot the ROC.
    
    Arguments:
    
        fpr:   false positive rates,
        tpr:   true positive rates,
        score: AUC.
    '''
    
    if ax is None:
        
        _, ax = subplots()

    sns.lineplot(x=fpr,
                 y=tpr,
                 ci=None,
                 ax=ax
                )

    sns.lineplot(x=np.linspace(0.0, 1.0, num=100),
                 y=np.linspace(0.0, 1.0, num=100),
                 color='tab:red',
                 linestyle='--',
                 ax=ax
                )
    
    ax.set(xlabel='false positive rate',
           ylabel='true positive rate',
           title=f'Receiver Operating Characteristic (area: {score:.3f})'
          )
    
    plt.tight_layout()
