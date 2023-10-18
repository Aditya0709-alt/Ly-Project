import os
from PIL import Image

def split_tif_images_with_overlap(input_folder, output_folder):
    # List all files in the input folder
    tif_files = [file for file in os.listdir(input_folder) if file.lower().endswith('.tif')]
    
    # Process each TIFF image
    for file_name in tif_files:
        input_path = os.path.join(input_folder, file_name)
        output_path1 = os.path.join(output_folder, f'left_half_{file_name}')
        output_path2 = os.path.join(output_folder, f'right_half_{file_name}')
        
        # Open the TIFF image
        image = Image.open(input_path)
        
        # Get the width and height of the image
        width, height = image.size
        
        # Calculate the middle point of the image
        middle = width // 2
        
        # Calculate overlap width (50% of the width)
        overlap_width = middle // 2
        
        # Split the image with 50% overlap
        left_half = image.crop((0, 0, middle + overlap_width, height))
        right_half = image.crop((middle - overlap_width, 0, width, height))
        
        # Save the split halves as separate images
        left_half.save(output_path1)
        right_half.save(output_path2)

# Example usage
input_folder_path = 'path'
output_folder_path = 'path'

split_tif_images_with_overlap(input_folder_path, output_folder_path)
