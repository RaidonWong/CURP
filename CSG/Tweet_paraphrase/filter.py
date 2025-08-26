import json

def extract_and_combine_data(csg_news_path, val_final_path, output_path):
    """
    Extracts 'prediction' and 'output' attributes from two JSON files,
    combines them, and saves the result to a new JSON file.

    Args:
        csg_news_path (str): The file path to the csg_news.json file.
        val_final_path (str): The file path to the val_final.json file.
        output_path (str): The file path for the new output JSON file.
    """
    combined_data = []

    try:
        # Load data from both files
        with open(csg_news_path, 'r', encoding='utf-8') as f:
            csg_news_data = json.load(f)

        with open(val_final_path, 'r', encoding='utf-8') as f:
            val_final_data = json.load(f)

        # Check if the number of data points is the same
        if len(csg_news_data) != len(val_final_data):
            print("Warning: The number of data points in the two files do not match. Processing the smaller number.")
            num_data = min(len(csg_news_data), len(val_final_data))
        else:
            num_data = len(csg_news_data)

        # Iterate and combine the data
        for i in range(num_data):
            prediction = csg_news_data[i].get('prediction')
            output = val_final_data[i].get('output')

            # Only add to the list if both attributes exist
            if prediction is not None and output is not None:
                combined_data.append({
                    'prediction': prediction,
                    'gold': output
                })

        # Save the combined data to the new file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)

        print(f"Successfully extracted and saved {len(combined_data)} data points to '{output_path}'.")

    except FileNotFoundError as e:
        print(f"Error: One of the files was not found. Please check the paths. Details: {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from one of the files. Please check if the files are valid JSON. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Define the file paths
csg_news_file = '/root/wangliang/Understanding-Generation/curp/CSG/csg_tweet.json'
val_final_file = '/root/wangliang/Understanding-Generation/curp/CSG/Tweet_paraphrase/val_final.json'
output_file = '/root/wangliang/Understanding-Generation/curp/CSG/Tweet_paraphrase/filtered_ans.json'

# Run the function
extract_and_combine_data(csg_news_file, val_final_file, output_file)