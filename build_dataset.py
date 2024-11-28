import json
import pandas as pd
from rich import print

# Load local Parquet dataset
parquet_file_path = './instruction_dataset.parquet'
dataset = pd.read_parquet(parquet_file_path)

# Print the columns of the dataset to verify
print("Dataset Columns:", dataset.columns)

# Convert dataset to list of dictionaries
json_data = dataset.to_dict(orient='list')

# Print the first row to verify
print("First Row:", {key: json_data[key][0] for key in json_data})

# Initialize list to store Alpaca format data
alpaca_data = []

# Process each entry in the dataset
for i in range(len(json_data['instruction'])):
    instruction = json_data['instruction'][i]
    input_text = json_data['input'][i]
    original_output = json_data['output'][i]

    # Create Alpaca format entry
    alpaca_entry = {
        "instruction": instruction,
        "input": input_text,
        "output": original_output
    }
    alpaca_data.append(alpaca_entry)

# Save in Alpaca format
with open("alpaca_dataset.json", "w") as file:
    json.dump(alpaca_data, file, indent=4)

print("Alpaca format dataset saved as 'alpaca_dataset.json'")
