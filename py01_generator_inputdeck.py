# %%
import random
import os

# Function to generate a random input file
def generate_input_deck():
    # Randomly choose MAT_E_MODUL and corresponding MAT_V_ZAHL
    mat_e_modul = random.choice([210000, 70000])
    mat_v_zahl = 0.30 if mat_e_modul == 210000 else 0.35

    # Generate a random LOAD_F between -10000 and -100 with step of 100
    load_f = random.randrange(-10000, -100, 100)

    # Template for the input deck
    input_deck = f"""*HEADING
Simple beam with 'clamped-clamped' configuration and loaded in the middle
**
*NODE
1,  0.0,  0.0, 0.0
2,  1.0,  0.0, 0.0
3,  2.0,  0.0, 0.0
4,  3.0,  0.0, 0.0
5,  4.0,  0.0, 0.0
6,  5.0,  0.0, 0.0
7,  6.0,  0.0, 0.0
8,  7.0,  0.0, 0.0
9,  8.0,  0.0, 0.0
10, 9.0,  0.0, 0.0
11, 10.0, 0.0, 0.0
**
*ELEMENT, TYPE=B31, ELSET=ELEMS_ALL
1, 1, 2
2, 2, 3
3, 3, 4
4, 4, 5
5, 5, 6
6, 6, 7
7, 7, 8
8, 8, 9
9, 9, 10
10, 10, 11
**
*MATERIAL, NAME=MATERIAL
*ELASTIC
{mat_e_modul}, {mat_v_zahl}
**
*BEAM SECTION, ELSET=ELEMS_ALL, MATERIAL=MATERIAL, SECTION=RECT
1.0, 1.0
**
*NSET, NSET=N_FIX
1, 11
**
*NSET, NSET=N_ALL
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
**
*STEP
*STATIC
*BOUNDARY
N_FIX, 1, 6  
**
*CLOAD
6, 2, {load_f}
*NODE PRINT, NSET=N_ALL
U
*END STEP
"""

    # Define the file_name according to the generated variables
    file_name = os.path.join(output_dir, f"inputdeck_EMODUL_{mat_e_modul}_LOAD_{abs(load_f)}.inp")

    # Write the input deck to a file
    with open(file_name, 'w') as file:
        file.write(input_deck)

    return file_name

# Generate multiple input decks
def generate_multiple_decks(num_decks, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(1, num_decks + 1):
        file_name = generate_input_deck()
        print(f"{i} -> Generated: {file_name}")

# Parameters
num_decks = 1000  # Number of input decks to generate
output_dir = "inputdecks"  # Output directory

# Generate the input decks
generate_multiple_decks(num_decks, output_dir)
