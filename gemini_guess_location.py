"""This file loops through an input folder and passes each image to Gemini 2.5 Pro.
 The model attempts to identify the image's location. Results are saved to results.csv"""
import os
import time
from google.api_core import exceptions
import google.generativeai as genai
from PIL import Image
import pandas as pd

#Specifying Input folder, prompt, and api key constant variables
INPUT_FOLDER = "output_images_B"
PROMPT = """Please identify the specific location that this image was taken.
Provide as much information behind your thought process as possible. 
Make the last line of your response be the specific location"""
API_KEY = ""

#Configure model parameters to make it slightly random
generation_config = {
  "temperature": 0.2,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}

#Pass in api key
genai.configure(api_key=API_KEY)

#Create model. We're using gemini 2.5 pro, if that his limit, can use 2.5 flash
model = genai.GenerativeModel(
    model_name="gemini-2.5-pro",
    generation_config=generation_config,
)

#Set up array to collect responses
responses = []

#Putting the prompting in a try-except to catch if we get rate-limited early
try:
    for filename in os.listdir(INPUT_FOLDER): #Loop through all files in the input folder
        full_path = os.path.join(INPUT_FOLDER, filename)
        print(filename)
        img = Image.open(full_path) #Open as a PILLOW image

        response = model.generate_content([PROMPT, img]) #Prompt the model
        responses.append({
            "image": filename,
            "response": response.text})
        time.sleep(5) #Prevents hitting the 15/min rate limit
    resultsdf = pd.DataFrame(responses)
    resultsdf.to_csv("results_25pro.csv", index=False)
except exceptions.ResourceExhausted: #If we get rate-limited, save results so far
    print("Daily rate limit hit. Saving results so far...")
    resultsdf = pd.DataFrame(responses)
    resultsdf.to_csv("results_25pro.csv", index=False)
