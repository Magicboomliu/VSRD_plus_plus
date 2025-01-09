import os
from PIL import Image

# Parameters
type='bev'
root_folder =  "/media/zliu/data12/TPAMI_Results/Figures_For_Papers/Ablations/vsrd_velocity/seq1/"


folder_path = os.path.join(root_folder,type)  # Replace with your folder path
output_gif_path = os.path.join(root_folder,type+".gif")
  # Output GIF file name
fps = 3  # Frames per second

# Collect all image files in the folder
image_files = sorted(
    [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('png', 'jpg', 'jpeg', 'bmp'))]
)

# Ensure there are images to process
if not image_files:
    raise ValueError("No image files found in the specified folder!")

# Load images
frames = [Image.open(img) for img in image_files]

# Create a GIF
frames[0].save(
    output_gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=1000 // fps,  # Duration per frame in milliseconds
    loop=0  # Infinite loop
)

print(f"GIF saved as '{output_gif_path}'")
