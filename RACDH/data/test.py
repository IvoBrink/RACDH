import pandas as pd
import json  # Import the json module

# Read the CSV file into a DataFrame
data = pd.read_csv("RACDH/data/nounlist.csv", header=None)

# Convert the DataFrame to a NumPy array
array_data = data.to_numpy()

# Convert the array to a list and then to JSON
json_data = array_data.flatten().tolist()  # Flatten and convert to list

# Save the JSON data to a file
with open("RACDH/data/nouns.json", "w") as json_file:  # Specify the output file path
    json.dump(json_data, json_file, indent=4)  # Write the JSON data to the file
