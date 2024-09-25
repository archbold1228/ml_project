# %%
import os
import re
import collections
import pandas as pd

# Directory containing the .dat files
data_dir = "inputdecks"

# Function to parse a .dat file and extract the displacement values for nodes 1 to 11
def parse_dat_file(file_path):
    
    # Extract the E-modul and the applied load to the middle point out of the file_path
    emodul, load = re.search(r'inputdeck_EMODUL_(\d+)_LOAD_(\d+)\.dat', file_path).groups()

    # Add the E-modul and the applied in a list (these two variables will be the features in the ML-model)
    ml_database = [emodul, load]

    # Add the vertical displacement vy for each node of the inputdeck model (these will be the desired output in the ML-model)
    with open(file_path, 'r') as file:
        for line in file.readlines():
            # We're interested in lines that start with the node numbers (1 to 11)
            if line.strip() and line.split()[0].isdigit():
                node, vx, vy, vz = line.split()
                ml_database.append(float(vy))  # Extract only the vy (y-direction displacement)
    
    return ml_database

# Create an empty DataFrame to store the results
# E-modul, Load and Node_1 to Node_11 as column headers
columns = ["E_modul", "Load"] + [f"Node_{i}" for i in range(1, 12)]  
data = pd.DataFrame(columns=columns)

# Create a list to store all rows
rows = []

# Iterate through all .dat files in the directory
for file_name in os.listdir(data_dir):
    if file_name.endswith(".dat"):
        file_path = os.path.join(data_dir, file_name)
        ml_database = parse_dat_file(file_path)

        # Remove the '.dat' suffix from the file_name 
        file_base_name = os.path.splitext(file_name)[0]
        
        # Append the data to the rows list (each file is a new row)
        if len(ml_database) == len(columns):  # Ensure we have all the data we need for the ML-model
            row = pd.Series(ml_database, index=columns, name=file_base_name)
            rows.append(row)

# Concatenate all rows into the DataFrame at once
data = pd.concat(rows, axis=1).T

# Reset the index to clean up the DataFrame
data.reset_index(inplace=True)
data.rename(columns={'index': 'Sample'}, inplace=True)

# Display the DataFrame to the user
data.to_csv("./inputdecks/ml_database.csv", index=False)
