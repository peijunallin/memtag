from datasets import load_dataset
import os
import json
target_dir = 


# Create the directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)
# Download the dataset with cache_dir parameter



        
#-----------------------------------------------------------------------
musiq = load_dataset("dgslibisey/MuSiQue", cache_dir=target_dir)['train'] 
# Dataset({
#     features: ['id', 'paragraphs', 'question', 'question_decomposition', 'answer', 'answer_aliases', 'answerable'],
#     num_rows: 2417
# })
processed_musiq = []

for example in musiq:
    # Concatenate all paragraphs with their titles
    context = ""
    for para in example['paragraphs']:
        title = para['title']
        text = para['paragraph_text']
        context += f"{title}: {text} "

    # Create a new example with the concatenated context
    processed_example = {
        'context': context.strip(),
        'question': example['question']
    }
    processed_musiq.append(processed_example)



# # Convert to a Dataset object if needed
# from datasets import Dataset
# processed_train_dataset = Dataset.from_list(processed_train)
# # Save the processed dataset if needed
# processed_train_dataset.save_to_disk("processed_musique_train")
# # Example of the first processed item
# print(processed_train[0]['context'][:500])

#-----------------------------------------------------------------------
with open('datasets/all_rawdata/hotpot_train_v1.1.json', 'r', encoding='utf-8') as file:
    hotpot = json.load(file)
    

processed_hotpot = []
for example in hotpot:
    # Concatenate all paragraphs with their titles
    context = ""
    for para in example['context']:
        title = para[0]
        text = ''.join(para[1])
        context += f"{title}: {text} "

    # Create a new example with the concatenated context
    processed_example = {
        'context': context.strip(),
        'question': example['question']
    }
    processed_hotpot.append(processed_example)
    
#-----------------------------------------------------------------------
with open('2wikimultihopqa/train.json', 'r', encoding='utf-8') as file:
    wiki = json.load(file)
processed_wiki = []
for example in wiki:
    # Concatenate all paragraphs with their titles
    context = ""
    for para in example['context']:
        title = para[0]
        text = ''.join(para[1])
        context += f"{title}: {text} "

    # Create a new example with the concatenated context
    processed_example = {
        'context': context.strip(),
        'question': example['question']
    }
    processed_wiki.append(processed_example)
    
    
#-----------------------------------------------------------------------
    





import pandas as pd
import csv


                

musiq_with_source = [{'source': 'musiq', 'context': item['context'], 'question': item['question']} 
                    for item in processed_musiq]

hotpot_with_source = [{'source': 'hotpot', 'context': item['context'], 'question': item['question']} 
                     for item in processed_hotpot]

wiki_with_source = [{'source': 'wiki', 'context': item['context'], 'question': item['question']} 
                   for item in processed_wiki]



# Merge all datasets
[REDACTED]

# Create a DataFrame
df = pd.DataFrame(merged_data)

# Define the output path
output_path = 'merged_datasets.csv'

# Save to CSV
# Using pandas with appropriate settings for large text fields
df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')

# Print some stats
print(f"Total examples: {len(merged_data)}")
print(f"Examples per dataset:")
print(f"  MuSiQue: {len(musiq_with_source)}")
print(f"  HotpotQA: {len(hotpot_with_source)}")
print(f"  2WikiMultiHopQA: {len(wiki_with_source)}")
print(f"CSV file saved to: {output_path}")









