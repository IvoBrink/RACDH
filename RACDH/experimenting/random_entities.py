import os
import json

def merge_unique_names(directory):
    unique_names = set()
    
    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.startswith("data-") and filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            
            # Open and read the JSON file
            with open(filepath, 'r') as file:
                data = json.load(file)
                for entry in data:
                    unique_names.add(entry['name'])
    
    # Convert the set to a list
    unique_names_list = list(unique_names)
    print(f"The number of unique names: {len(unique_names_list)}")
    
    # Save the unique names to a new JSON file
    with open(os.path.join(directory, 'merged_unique_names.json'), 'w') as outfile:
        json.dump(unique_names_list, outfile, indent=4)

# Specify the directory containing the JSON files
directory_path = 'RACDH/data'
merge_unique_names(directory_path)
