#####################################################################
#
# Setup Folders for Collection
#
#####################################################################

import os
import numpy as np
from globals import *

new_action_folder_count = 0 # Initialize counter for new action folders

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path): # Check if the action folder already exists
        new_action_folder_count += 1 # Increment the new action folder counter
        for sequence in range(1,no_sequences+1):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass
    else:
        continue

# Report back how many folders were made
if new_action_folder_count == 0:
    print("No new action folders were made.")
else:
    print(f"Successfully created {new_action_folder_count} new action folders.")