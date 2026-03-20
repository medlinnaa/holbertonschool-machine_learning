#!/usr/bin/env python3
"""7-early_stopping.py"""

def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines whether to stop training early based on validation cost.

    Parameters:
    - cost (float): current validation cost
    - opt_cost (float): lowest recorded validation cost
    - threshold (float): minimum improvement considered meaningful
    - patience (int): maximum allowed epochs without meaningful improvement
    - count (int): current number of epochs without meaningful improvement

    Returns:
    - stop (bool): True if training should stop, False otherwise
    - updated_count (int): updated count of epochs without improvement
    """
    if opt_cost - cost > threshold:
        # Improvement is meaningful = reset counter
        count = 0
    else:
        # No meaningful improvement = increment counter
        count += 1

    # Stop training if patience is exceeded
    stop = count >= patience

    return stop, count
