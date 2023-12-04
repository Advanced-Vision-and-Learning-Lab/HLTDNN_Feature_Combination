import os
import csv
import re

def extract_accuracy_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            match = re.search(r'Average accuracy: (\d+\.\d+) Std: (\d+\.\d+)', content)
            if match:
                return float(match.group(1)), float(match.group(2))
            else:
                return None, None
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None, None

def process_subfolders(main_folder, output_csv):
    if not os.path.exists(main_folder):
        print(f"Error: Main folder '{main_folder}' not found.")
        return

    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Counter', 'Subfolder', 'Average Accuracy', 'Standard Deviation'])

        counter = 1
        for root, dirs, files in os.walk(main_folder):
            for file in files:
                if file == 'Overall_Accuracy.txt':
                    folder_path = os.path.join(root, file)
                    accuracy, std = extract_accuracy_from_file(folder_path)
                    if accuracy is not None and std is not None:
                        subfolder_name = os.path.basename(root)
                        csv_writer.writerow([counter, subfolder_name, accuracy, std])
                        counter += 1

        if csv_file.tell() == 0:
            print("Warning: No 'Overall_Accuracy.txt' files found in sub-folders.")

def get_top_accuracies(input_csv, output_csv, top_n=6):
    try:
        with open(input_csv, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            rows = list(csv_reader)

            # Sort rows based on 'Average Accuracy' in descending order
            sorted_rows = sorted(rows, key=lambda x: float(x['Average Accuracy']), reverse=True)

            with open(output_csv, 'w', newline='') as output_file:
                fieldnames = csv_reader.fieldnames
                csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                csv_writer.writeheader()
                for row in sorted_rows[:top_n]:
                    csv_writer.writerow(row)

    except Exception as e:
        print(f"Error processing input CSV '{input_csv}': {e}")

if __name__ == "__main__":
    main_folder_path = '/home/ugrads/i/imasabo2k18/peeplesLab/HLTDNN_Feature_Combination/Saved_Models/Fine_Tuning/DeepShip/HistTDNN_16/Parallel'
    output_csv_path = 'output.csv'
    best_accuracies_csv_path = 'best_accuracies.csv'

    process_subfolders(main_folder_path, output_csv_path)
    print(f"CSV file '{output_csv_path}' has been created.")

    get_top_accuracies(output_csv_path, best_accuracies_csv_path)
    print(f"CSV file '{best_accuracies_csv_path}' has been created with the top 6 accuracies.")
