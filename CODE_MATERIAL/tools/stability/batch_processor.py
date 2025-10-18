#!/usr/bin/env python3
"""
Runtime Stability Tools - Batch Processing and Checkpoint Recovery
"""

import os
import json
import argparse
from pathlib import Path

def create_checkpoint(checkpoint_file, processed_items):
    """Save checkpoint file"""
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(list(processed_items), f, indent=2)

def load_checkpoint(checkpoint_file):
    """Load checkpoint file"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def process_in_batches(items, batch_size, process_func, checkpoint_file=None, checkpoint_interval=100):
    """Process items in batches with checkpointing"""
    processed = load_checkpoint(checkpoint_file) if checkpoint_file else set()
    total_items = len(items)
    
    for i in range(0, total_items, batch_size):
        batch = items[i:i + batch_size]
        batch_to_process = [item for item in batch if item not in processed]
        
        if not batch_to_process:
            print('Batch ' + str(i//batch_size + 1) + ': Already processed, skipping')
            continue
            
        print('Processing batch ' + str(i//batch_size + 1) + '/' + str(total_items//batch_size + 1) + ' (' + str(len(batch_to_process)) + ' items)')
        
        try:
            process_func(batch_to_process)
            processed.update(batch_to_process)
            
            # Save checkpoint periodically
            if checkpoint_file and (i + batch_size) % checkpoint_interval == 0:
                create_checkpoint(checkpoint_file, processed)
                print('Checkpoint saved: ' + str(len(processed)) + ' items processed')
                
        except Exception as e:
            print('Error processing batch: ' + str(e))
            if checkpoint_file:
                create_checkpoint(checkpoint_file, processed)
                print('Checkpoint saved before error: ' + str(len(processed)) + ' items processed')
            raise
    
    if checkpoint_file:
        create_checkpoint(checkpoint_file, processed)
        print('Final checkpoint saved: ' + str(len(processed)) + ' items processed')

if __name__ == "__main__":
    print("Runtime Stability Tools loaded")