# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 13:26:58 2025

@author: djwou
"""

import google.generativeai as genai
import os
import time
from PIL import Image
from google.api_core import exceptions

# --- Configuration ---
API_KEY = '' # Use the same billing-enabled key
INPUT_FOLDER = 'input_images'
OUTPUT_FOLDER = 'output_images'
BLUR_PROMPT = "Blur the image so the location is unidentifiable"
MAX_RETRIES = 5 # Max attempts per file
RETRY_WAIT_TIME = 60 # Seconds to wait

# --- Files to Retry ---
# I've added all the files from your log plus the 3 you mentioned.
# Feel free to add/remove from this list.
files_to_retry = [
    "kyungbook.webp"  # User-reported failure
]
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

print(f"--- Starting retry script for {len(files_to_retry)} images ---")

# Loop through ONLY the failed files
for filename in files_to_retry:
    
    image_path = os.path.join(INPUT_FOLDER, filename)
    
    # Check if file exists before trying
    if not os.path.exists(image_path):
        print(f"SKIPPING: Cannot find file {image_path}")
        continue

    processed = False
    retry_count = 0

    while not processed and retry_count < MAX_RETRIES:
        try:
            print(f"Processing: {image_path} (Attempt {retry_count + 1})")
            
            img = Image.open(image_path)
            response = model.generate_content([BLUR_PROMPT, img])
            
            output_image_data = response.candidates[0].content.parts[0].inline_data.data
            
            filename_no_ext = os.path.splitext(filename)[0]
            output_filename = f"blurred_{filename_no_ext}.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            with open(output_path, 'wb') as f:
                f.write(output_image_data)
            
            print(f"Successfully saved to: {output_path}")
            processed = True # Mark as done

        except (exceptions.ResourceExhausted, exceptions.InternalServerError) as e:
            # --- THIS IS THE UPGRADE ---
            # Now catches Rate Limit AND 500 Internal errors
            retry_count += 1
            print(f"Error for {image_path}: {e}")
            if retry_count < MAX_RETRIES:
                print(f"Waiting {RETRY_WAIT_TIME} seconds to retry...")
                time.sleep(RETRY_WAIT_TIME)
            
        except Exception as e:
            # This catches other errors (like corrupt image)
            print(f"Failed to process {image_path} with non-retryable error: {e}")
            processed = True # Give up and skip this file

    if not processed:
        print(f"--- FAILED TO PROCESS {image_path} after {MAX_RETRIES} attempts. ---")

print("\nBatch retry complete.")