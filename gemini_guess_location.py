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

def save_results(responses1):
    """Function to save results in a labelled dataframe, 
    prevents duplicating code for each except branch"""
    df = pd.DataFrame(responses1)
    if model.model_name=="models/gemini-2.5-pro":
        df.to_csv("results_25pro.csv", index=False)
    elif model.model_name=="models/gemini-2.5-flash":
        df.to_csv("results_25flash.csv", index=False)
    elif model.model_name=="models/gemini-2.0-flash":
        df.to_csv("results_20flash.csv", index=False)
    else:
        df.to_csv("resultsOTHERMODEL.csv", index=False)
    print("Saved results")

#Configure model parameters to make it slightly random
generation_config = {
  "temperature": 0.2,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}

#Pass in api key
genai.configure(api_key=API_KEY)

#Create model. We're using gemini 2.5 pro, if that hits limit, can use 2.5 flash
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
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
        if model.model_name=="models/gemini-2.5-pro":
            print("pro 2.5 - sleeping 20 seconds")
            time.sleep(20) #Prevents hitting the 2/min rate limit
        elif model.model_name=="models/gemini-2.0-flash":
            print("flash 2.0 - sleeping 4.5 seconds")
            time.sleep(4.5) #Prevents hitting the 15/min rate limit
        else:
            print("unknown model - sleeping 15 seconds")
            time.sleep(15)
        #Flash 2.5 is so slow we don't need to sleep anything
        #elif model.model_name=="models/gemini-2.5-flash":
        #    print("flash 2.5 - sleeping 1 second")
        #    time.sleep(1) #Prevents hitting 10/min rate limit
    save_results(responses)
except exceptions.ResourceExhausted: #If we get rate-limited, save results so far
    print("Daily rate limit hit. Saving results so far...")
    save_results(responses)
except exceptions.InvalidArgument:
    print("Invalid API key")
except KeyboardInterrupt:
    save_results(responses)
    print("User stopped program")
save_results(responses)
