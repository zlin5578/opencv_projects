import json
import os
import argparse

def convert_labelme_folder_to_csv(input_folder):
    # Define output folder name based on input folder
    output_folder = f"{input_folder}_csv"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(input_folder, filename)
            csv_filename = f"{os.path.splitext(filename)[0]}.csv"
            csv_path = os.path.join(output_folder, csv_filename)

            # Load JSON file
            with open(json_path, 'r') as file:
                data = json.load(file)

            # Extract shapes (bounding boxes)
            shapes = data.get('shapes', [])
            num_boxes = len(shapes)

            # Open the CSV file for writing
            with open(csv_path, 'w') as csv_file:
                # Write the number of bounding boxes as the first line
                csv_file.write(f"{num_boxes}\n")

                # Iterate through the shapes and write bounding box coordinates
                for shape in shapes:
                    points = shape.get('points', [])
                    if len(points) == 2:
                        minX = min(points[0][0], points[1][0])
                        minY = min(points[0][1], points[1][1])
                        maxX = max(points[0][0], points[1][0])
                        maxY = max(points[0][1], points[1][1])
                        # Write the coordinates to the CSV file with space as separator
                        csv_file.write(f"{int(minX)} {int(minY)} {int(maxX)} {int(maxY)}\n")

# Main function to parse command-line arguments and call the conversion function
def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Convert LabelMe JSON files to CSV.')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Input folder containing JSON files.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the conversion function with the provided directory
    convert_labelme_folder_to_csv(args.directory)

if __name__ == "__main__":
    main()
