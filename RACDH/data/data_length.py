import json

def count_dataset_splits(base_path="MIND/auto-labeled/wiki/"):
    splits = ['wiki_train.json', 'wiki_valid.json', 'wiki_test.json']
    counts = {}
    
    for split in splits:
        try:
            with open(base_path + split, 'r') as f:
                data = json.load(f)
                counts[split] = len(data)
        except FileNotFoundError:
            print(f"Warning: {split} not found")
            counts[split] = 0
            
    return counts

if __name__ == "__main__":
    counts = count_dataset_splits()
    print("\nDataset split sizes:")
    for split, count in counts.items():
        print(f"{split}: {count} examples")
