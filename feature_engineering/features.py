#############################################################################################################################
# This is a python script to get features from LLC Access Traces generated using ChampSim
# These access traces give us pc, full memory address, cache set, cache way, type of access, recency of access 
# and whether the access was a hit or a miss for the LLC using LRU replacement policy
#
# We are extracting the following features from them:
# 1) Tag of memory address for current address
# 2) Block offset (64B) of memory address for current address
# 3) Type of Access
# 4) Recency of Access
# 5) Three 3-bit long subsets of last tag bits
# 6) Line hits since insertion - number of times the cache line was hit since insertion (i.e, the last miss)
# 7) Sets accessed since last miss - number of times the current set was accessed since the cache experienced a miss
############################################################################################################################

import os
import numpy as np
import pandas as pd

# Main directory containing the CSV files
main_directory = "./"

# Define a function to calculate the tag
def get_tag(address):
    tag_hex = hex(int(address,16) >> 17)
    return tag_hex

def get_offset(address):
    offset = hex(int(address,16)&0x3f)
    return offset

def get_3tag_1(tag):
    three_tag = hex(int(tag,16) & 0x7) 
    return three_tag

def get_3tag_2(tag):
    three_tag = hex((int(tag,16) & 0x38) >> 3) 
    return three_tag

def get_3tag_3(tag):
    three_tag = hex((int(tag,16) & 0x1c0)>> 6)
    return three_tag

# Process each CSV file
for file_name in os.listdir(main_directory):
    if file_name.endswith('.csv'):
        # Prevent reading csv fiels with extracted features
        if file_name.endswith('_features.csv'):
            break
        file_path = os.path.join(main_directory, file_name)
        output_filename = os.path.join(main_directory, os.path.splitext(file_name)[0] + '_features.csv')
        print(f"Processing file: {file_path}")

        # Read CSV file and add column names for readability
        df = pd.read_csv(file_path, header=None)
        df.columns = ['pc', 'address', 'set', 'way', 'instr_type', 'recency', 'hit/miss'] #, 'hits_count', 'preuse_dist']

        # Get tag column
        df['tag'] = df['address'].apply(get_tag)
        address_index = df.columns.get_loc('address')  # Get the index of the address column
        df.insert(address_index + 1, 'tag', df.pop('tag'))  # Insert the tag column after the address column

        # Get offset column
        df['offset'] = df['address'].apply(get_offset)
        address_index = df.columns.get_loc('address')  # Get the index of the address column
        df.insert(address_index + 2, 'offset', df.pop('offset'))  # Insert the tag column after the address column

        # Get 3 bit subsets of tag
        df['3_tag_1'] = df['tag'].apply(get_3tag_1)
        df['3_tag_2'] = df['tag'].apply(get_3tag_2)
        df['3_tag_3'] = df['tag'].apply(get_3tag_3)

        # Initialize results lists
        hits_count = []
        salm = []

        # Iterate through the DataFrame
        for i in range(df.shape[0]):
            current_tag = df.loc[i, 'tag']
            current_set = df.loc[i, 'set']

            # Initialize counters
            hit_count = 0
            salm_count = 0
            
            # Calculate hits since last insertion
            for j in range(i - 1, -1, -1):
                if df.loc[j, 'tag'] == current_tag:
                    if df.loc[j, 'hit/miss'] == 1:
                        hit_count += 1
                    else:
                        break
            #Append
            hits_count.append(hit_count)

            # Calculated Sets accessed since last miss (SALM)
            for j in range(i - 1, -1, -1):
                if df.loc[j, 'hit/miss'] == 0:  
                        break
                else:
                    if df.loc[j, 'set'] == current_set:  
                        salm_count += 1 
            #Append
            salm.append(salm_count)

        # Add the results as new columns
        df['hits_count'] = hits_count
        df['salm'] = salm

        # Save the DataFrame to a new CSV file
        df.to_csv(output_filename, index=False)
        print(f"Processed file saved to: {output_filename}")