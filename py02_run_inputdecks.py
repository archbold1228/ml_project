# %% 
import os
import subprocess

# Directory where input decks are located
input_dir = "inputdecks"

# Docker command template (without .inp extension for file names)
docker_command = "docker run --rm --platform linux/amd64 -v ./inputdecks:/data calculix/ccx ccx /data/{}"

# Function to run all .inp files in the directory without including the '.inp' extension in the command
def run_input_decks(input_dir):
    # Ensure the directory exists
    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' does not exist.")
        return

    # Loop over all files in the directory
    for file_name in os.listdir(input_dir):
        # Only consider files with the .inp extension
        if file_name.endswith(".inp"):
            # Remove the .inp extension for the Docker command
            file_base_name = os.path.splitext(file_name)[0]
            print(f"Running simulation for {file_base_name}...")

            # Build the full Docker command with the base file name (without .inp)
            command = docker_command.format(file_base_name)
            
            # Execute the Docker command
            try:
                subprocess.run(command, shell=True, check=True)
                print(f"Simulation completed for {file_base_name}.")
            except subprocess.CalledProcessError as e:
                print(f"Error running simulation for {file_base_name}: {e}")

# Run the function to execute all .inp files in the 'inputdecks' directory
run_input_decks(input_dir)
