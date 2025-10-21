import google.generativeai as genai
import os
import glob
import time
from PIL import Image
from google.api_core import exceptions

# --- Configuration ---
API_KEY = ''
INPUT_FOLDER = 'input_images'
OUTPUT_FOLDER = 'output_images'
BLUR_PROMPT = "Blur the image so the location is unidentifiable"
# --- End of Configuration ---

# Configure the API client
genai.configure(api_key=API_KEY)

# Set up the model
generation_config = {
  "temperature": 0.4,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-image",
    generation_config=generation_config
)

# Create the output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created output folder: {OUTPUT_FOLDER}")

# Find all .jpg, .png, and .webp images in the input folder
image_paths = glob.glob(os.path.join(INPUT_FOLDER, '*.jpg'))
image_paths.extend(glob.glob(os.path.join(INPUT_FOLDER, '*.png')))
image_paths.extend(glob.glob(os.path.join(INPUT_FOLDER, '*.webp'))) # <-- SYNTAX FIX HERE

if not image_paths:
    print(f"No .jpg, .png, or .webp images found in '{INPUT_FOLDER}'.")
    exit()

print(f"Found {len(image_paths)} images to process...")

# Loop through each image and process it
for image_path in image_paths:
    
    processed = False
    while not processed:
        try:
            print(f"Processing: {image_path}")
            
            # Open the image (Pillow handles JPG, PNG, WebP automatically)
            img = Image.open(image_path)
            
            # Send the image and prompt to the API
            response = model.generate_content([BLUR_PROMPT, img])
            
            # Correct path to the image data
            output_image_data = response.candidates[0].content.parts[0].inline_data.data
            
            # Determine the output filename
            base_filename = os.path.basename(image_path)
            filename_no_ext = os.path.splitext(base_filename)[0]
            output_filename = f"blurred_{filename_no_ext}.png" # Save as PNG
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            # Save the new image
            with open(output_path, 'wb') as f:
                f.write(output_image_data)
            
            print(f"Successfully saved to: {output_path}")
            processed = True # Mark as done and exit the 'while' loop

        except exceptions.ResourceExhausted as e:
            # This is the rate limit error
            print(f"Rate limit hit for {image_path}. Waiting 60 seconds to retry...")
            time.sleep(60)
            
        except Exception as e:
            # This will catch other errors
            print(f"Failed to process {image_path} with a different error: {e}")
            processed = True # Mark as done to skip this failed image

print("\nBatch processing complete.")