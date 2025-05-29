import os
import csv
import argparse

def get_all_from_log(input_file: str):
    with open(input_file, 'r') as file:
        all_lines = [l for l in file if len(l.split('|')) > 3]

    test_ppl_10 = None
    data_dir = None
    model = None
    grammar = None
    div = None

    for line in all_lines:
        if 'epoch 010' in line and "valid" in line:
            fields = line.split('|')
            for field in fields:
                if 'ppl' in field:
                    test_ppl_10 = field.strip().split()[-1]
                    break

        if 'fairseq.data.data_utils' in line:
            data_dir = line.strip().split()[-1]

        if 'model:' in line:
            model = line.strip().split(':')[-1].strip()
            model = model.lower()
            if 'transformer' in model:
                model = 'transformer'
            elif 'lstm' in model:
                model = 'lstm'

    if not test_ppl_10 or not data_dir or not model:
        raise ValueError("Missing test perplexity at epoch 010, model, or data_dir.")

    parts = data_dir.strip().split('/')
    try:
        grammar = parts[2]
        div = parts[3].split('-')[0]
    except IndexError:
        raise ValueError(f"Cannot parse grammar/div from data_dir path: {data_dir}")

    return test_ppl_10, grammar, div, model

def iterate_over_folders(input_folder: str, output_folder: str):
    with open(os.path.join(output_folder, 'aggregated_ppl.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['grammar', 'div', 'model', 'ppl-10-epochs'])

        for file_name in os.listdir(input_folder):
            if file_name.endswith('.out'):
                input_path = os.path.join(input_folder, file_name)
                try:
                    ppl_10, grammar, div, model = get_all_from_log(input_path)
                    writer.writerow([grammar, div, model, ppl_10])
                except ValueError as e:
                    print(f"Error processing file {input_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract test PPL at epoch 010 from fairseq logs.")
    parser.add_argument("-i", "--input_file", type=str, required=True,
                        help="Path to input folder containing log.txt files")
    parser.add_argument("-O", "--output_folder", type=str, required=True,
                        help="Directory to save the output CSV")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    iterate_over_folders(args.input_file, args.output_folder)
    print(f"Aggregated test PPL saved to {os.path.join(args.output_folder, 'aggregated_ppl.csv')}")