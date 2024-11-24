import logging

from pathlib import Path 
import sys
import datetime

import functools
"""
Class to create all of the logging functionalities for documentation and debugging purposes

I use this logging module because I want to have a comprehensive way on interfacing the actions that I am doing in a complex system 
without ever having to read the code.
"""


# note that this function returns a ocnsistent metrics object that I should use
def add_performance_logging(logger, level, type = 'train'): 
    def decorator(func): 
        @functools.wraps(func)
        def new_function(self, *args, **kwargs):
            metrics, hard_cases = func(self, *args, **kwargs) 

            message = f'''
            Epoch {metrics['epoch'] if type == 'train' else type} Summary:
            Average Loss: {metrics['loss']:.4f}
            Accuracy: {metrics['accuracy']:.2f}%
            Positive predictions: {metrics['pos_pred_ratio']:.2f}%
            Recall - 1: {metrics['recall - 1']:.4f}, 
            Precision - 1: {metrics['precision - 1']:.4f}
            Recall - 0: {metrics['recall - 0']:.4f}, 
            Precision - 0: {metrics['precision - 0']:.4f}
            '''

        

            logger.log(level, message)
            self.history[type].append(metrics)


            return metrics, hard_cases
            # Note here that the decorator modifies the fucntion byu removing hte fucntionality

        return new_function

    return decorator



def setup_logger(name='ml_experiment', log_dir='logs', level=logging.INFO):
    """
    Step-by-step logger setup with examples of what each component achieves
    """
    # STEP 1: Create the base logger
    logger = logging.getLogger(name)
    logger.setLevel(level)  # Base minimum level for ALL handlers
    
    # Example of why hierarchical naming is important:
    #   ml_experiment.training
    #   ml_experiment.evaluation
    #   ml_experiment.preprocessing
    
    # STEP 2: Setup Log Directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    # Important for: Organizing logs by experiment/run
    
    # STEP 3: Create Formatters
    # Detailed formatter for debugging and error analysis
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Example output:
    # "2024-11-14 10:23:45,123 - ml_experiment - ERROR - Model failed to converge"
    
    # Simple formatter for console viewing
    simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
    # Example output:
    # "INFO - Starting training epoch 1"
    
    # STEP 4: Create File Handler (Debug and above)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(log_path / f'experiment_{timestamp}.log')
    fh.setLevel(logging.DEBUG)  # Captures everything
    fh.setFormatter(detailed_formatter)
    # Purpose: Complete historical record with timestamps
    # Example debug message:
    # "2024-11-14 10:23:45,123 - ml_experiment - DEBUG - Batch 1: loss=0.534"
    
    # STEP 5: Create Console Handler (Info and above)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)  # Less verbose for console
    ch.setFormatter(simple_formatter)
    # Purpose: Real-time monitoring without clutter
    # Example console output:
    # "INFO - Epoch 1/10: accuracy=0.856"
    
    # STEP 6: Create Error Handler (Error and above)
    eh = logging.FileHandler(log_path / f'errors_{timestamp}.log')
    eh.setLevel(logging.ERROR)  # Only serious issues
    eh.setFormatter(detailed_formatter)
    # Purpose: Separate error tracking
    # Example error log:
    # "2024-11-14 10:23:45,123 - ml_experiment - ERROR - Memory allocation failed"
    
    # STEP 7: Add all handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.addHandler(eh) # when an event is logged, we want to handle it in multipel ways not just logging --> how can we precisely in an event do we document     
    # define the output file, define the format, define which events we want.
    
    return logger