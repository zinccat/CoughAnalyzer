import os
from collections import defaultdict

def compute_label_distribution(folder_path, label_type):
    """
    Computes the label distribution from all files in the given folder.
    
    Parameters:
        folder_path (str): The path to the folder containing label files.
        label_type (str): The type of label ('respiratory_condition', 'fever_muscle_pain', 'status', or 'gender').
    
    Returns:
        dict: A dictionary where keys are labels (int) and values are counts.
    """
    label_distribution = defaultdict(int)

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Process only files (skip directories)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                content = f.read().strip()

            # Determine label from file content
            if content == "":
                label = 0
            else:
                parts = content.split(',')
                if label_type == 'respiratory_condition':
                    # Use parts[0]: 'False' -> 0, 'True' -> 1
                    if parts[0] == 'False':
                        label = 0
                    elif parts[0] == 'True':
                        label = 1
                    else:
                        label = 0
                elif label_type == 'fever_muscle_pain':
                    # Use parts[1]: 'False' -> 0, 'True' -> 1
                    if len(parts) > 1:
                        if parts[1] == 'False':
                            label = 0
                        elif parts[1] == 'True':
                            label = 1
                        else:
                            label = 0
                    else:
                        label = 0
                elif label_type == 'status':
                    # Use parts[2]: 'healthy' -> 0, 'symptomatic' -> 1, 'COVID-19' -> 2
                    if len(parts) > 2:
                        status = parts[2].lower()
                        if status == 'healthy':
                            label = 0
                        elif status == 'symptomatic':
                            label = 1
                        elif status in ['covid-19', 'covid']:
                            label = 2
                        else:
                            label = 0
                    else:
                        label = 0
                elif label_type == 'gender':
                    # Use parts[3]: 'male' -> 0, 'female' -> 1, 'other' -> 2
                    if len(parts) > 3:
                        gender = parts[3].strip().lower()
                        if gender == 'male':
                            label = 0
                        elif gender == 'female':
                            label = 1
                        elif gender == 'other':
                            label = 2
                        else:
                            raise ValueError(f"Unknown gender label '{gender}' in file {filename}.")
                    else:
                        raise ValueError(f"Insufficient data in file {filename} for gender label.")
                else:
                    raise ValueError(f"Unknown label type {label_type}.")

            # Update the count for the determined label
            label_distribution[label] += 1
    summary = sum(label_distribution.values())
    label_distribution = {k: v / summary * 100 for k, v in label_distribution.items()}
    return dict(label_distribution)


# Example usage:
if __name__ == "__main__":
    folder_path = "data/coughvid_labels/val"
    label_type = 'fever_muscle_pain'  # Example: 'respiratory_condition', 'fever_muscle_pain', 'status', or 'gender'
    distribution = compute_label_distribution(folder_path, label_type)
    print(f"{label_type}:", distribution)
