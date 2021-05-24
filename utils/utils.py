import sys
import tensorflow as tf

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
