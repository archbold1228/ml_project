ML-Driven Simulation Workflow with CalculiX and Python
This project automates the generation, execution, and collection of simulation results using CalculiX input decks, followed by training a machine learning model to predict simulation results.

Project Overview
This repository includes a full pipeline to:

Generate input decks for CalculiX simulations.
Run the generated input decks using a Docker container.
Collect and process the results of the simulations.
Train a machine learning model on the results to predict simulation outcomes.
The project consists of the following Python scripts:

py01_generator_inputdeck.py: Generates multiple input decks for simulations.
py02_run_inputdecks.py: Runs the generated input decks through CalculiX using Docker.
py03_collect_inputdecks_results.py: Collects and processes the results from the simulations.
py04_train_ml_model.py: Trains a machine learning model using the collected data from the simulations.
Prerequisites
Software Requirements:
Python 3.8+
Docker (for running the CalculiX simulations)
CalculiX (inside Docker, handled automatically)
Python Dependencies:
Install the required Python libraries via pip:

bash
Code kopieren
pip install -r requirements.txt
Ensure you have Docker installed and running on your system to execute the CalculiX simulations.

Project Structure
py01_generator_inputdeck.py: Generates random input decks for CalculiX.
py02_run_inputdecks.py: Executes the generated input decks via Docker.
py03_collect_inputdecks_results.py: Gathers the results from the completed simulations.
py04_train_ml_model.py: Trains a machine learning model on the simulation data to make predictions on new inputs.
Usage Guide
1. Generate Input Decks
The first step is to generate input decks for CalculiX simulations.

Run the py01_generator_inputdeck.py script:

bash
Code kopieren
python py01_generator_inputdeck.py
This script generates multiple .inp files based on random parameters like E-Modul, Poisson's ratio, and Load.

2. Run Input Decks
Next, run the generated input decks using CalculiX in Docker.

Run the py02_run_inputdecks.py script:

bash
Code kopieren
python py02_run_inputdecks.py
This script looks for all .inp files in the input decks directory, executes them using CalculiX via Docker, and generates output results.

3. Collect Results
Once the simulations are complete, gather the output results for processing.

Run the py03_collect_inputdecks_results.py script:

bash
Code kopieren
python py03_collect_inputdecks_results.py
This script collects the output displacements for each simulation and stores them in a CSV file for further processing.

4. Train Machine Learning Model
Finally, train a machine learning model using the collected simulation data.

Run the py04_train_ml_model.py script:

bash
Code kopieren
python py04_train_ml_model.py
This script trains a regression model to predict the displacements based on input parameters (E-Modul, Poisson's ratio, Load). The model can then be used to make predictions on new input sets without rerunning simulations.

File Descriptions
py01_generator_inputdeck.py
This script generates input decks for CalculiX based on randomized parameters:

E-Modul: Elastic modulus, which can vary between 210,000 and 70,000.
Load: Applied load in the simulation, which ranges between -10,000 and -100.
The generated input decks are saved in the inputdecks folder as .inp files.

py02_run_inputdecks.py
This script runs all generated .inp files through CalculiX inside a Docker container. It automatically mounts the inputdecks directory, executes the simulation, and saves the results.

py03_collect_inputdecks_results.py
This script parses the results of the CalculiX simulations (displacement values at specific nodes) and consolidates them into a CSV file. The results are then used for training the machine learning model.

py04_train_ml_model.py
This script trains a machine learning model (using PyTorch) to predict displacement values based on input parameters (E-Modul, Load). The model is optimized using Optuna to find the best hyperparameters. Once trained, the model can be used to make predictions on new data without needing to run simulations.

Example Workflow
Generate Input Decks:

bash
Code kopieren
python py01_generator_inputdeck.py
Run Simulations:

bash
Code kopieren
python py02_run_inputdecks.py
Collect Results:

bash
Code kopieren
python py03_collect_inputdecks_results.py
Train the Machine Learning Model:

bash
Code kopieren
python py04_train_ml_model.py
Results
The trained machine learning model is saved in the model.pth file, and the best hyperparameters used during training are stored in a JSON file best_10_hyperparams.json. Additionally, a CSV file with the predictions and error percentages is generated.

Contribution
If you would like to contribute to this project, feel free to fork the repository, create a feature branch, and submit a pull request.

Contact
For any inquiries, feel free to contact the project owner at archbold.alex@outlook.com
