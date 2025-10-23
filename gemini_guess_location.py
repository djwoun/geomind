from google.api_core import exceptions
import google.generativeai as genai
import os
from PIL import Image
import pandas as pd
import time

responses = []

INPUT_FOLDER = "output_images_B"
PROMPT = "Please identify the specific location that this image was taken. Provide as much information behind your thought process as possible. Make the last line of your response be the specific location"
API_KEY = ""
generation_config = {
  "temperature": 0.4,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-2.5-pro",
    generation_config=generation_config,
)

try:
    for filename in os.listdir(INPUT_FOLDER):
        full_path = os.path.join(INPUT_FOLDER, filename)
        print(filename)
        img = Image.open(full_path)

        response = model.generate_content([PROMPT, img])
        responses.append({
            "image": filename,
            "response": response.text})
        
        resultsdf = pd.DataFrame(responses)
        resultsdf.to_csv("results_25pro.csv", index=False)
except exceptions.ResourceExhausted:
    print("Daily rate limit hit. Saving results so far...")
    resultsdf = pd.DataFrame(responses)
    resultsdf.to_csv("results.csv", index=False)
