import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='notebook', palette='tab10')

def set_memory_growth():
    '''
    Set memory growth on GPU.
    '''
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        
        try:
            
            for gpu in gpus:
                
                tf.config.experimental.set_memory_growth(gpu, True)
        
        except RuntimeError as e:
            
            sys.stderr.write(e)

def subplots(nrows=1, ncols=1, figsize=(6,5)):
    '''
    Create plot grid.
    
    Returns:
        
        figure, axes.
    '''
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    plt.tight_layout()
    
    return fig, ax