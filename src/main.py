 #!/usr/bin/env python3

import flask
import os
import requests
import PIL
import PIL.ImageOps
import numpy as np
import json


app = flask.Flask(__name__)

predict_api_url = os.getenv('PREDICT_API_URL', 'http://localhost:8081/image')
model_url = os.getenv('MODEL_URL', 'NOT SET')

def log(e):
    print("{0}\n".format(e))

@app.route('/', methods=['GET'])
def index():
    return flask.render_template("mnist.html", model_url=model_url)

@app.route('/image', methods=['POST'])
def image():
    # Note, even though we do a little prep, we don't clean the image nearly
    # as well as the MNIST dataset expects, so there will be some issues with
    # generalizing it to handwritten digits

    # Start by taking the image into pillow so we can modify it to fit the
    # right size
    pilimage: PIL.Image.Image = PIL.Image.frombytes(
        mode="RGBA", size=(200, 200), data=flask.request.data)

    # Resize and process the image to fit the model's input size and format
    pilimage = pilimage.resize((20, 20))

    # Need to replace the Alpha channel, since 0=black in PIL
    newimg = PIL.Image.new(mode="RGBA", size=(20, 20), color="WHITE")
    newimg.paste(im=pilimage, box=(0, 0), mask=pilimage)

    # Turn it from RGB down to grayscale
    grayscaled_image = PIL.ImageOps.grayscale(newimg)

    # Add the padding so we have it at 28x28, with the 4px padding on all sides
    padded_image = PIL.ImageOps.expand(
        image=grayscaled_image, border=4, fill=255)

    # Invert the image because model expects black digits on white background
    inverted_image = PIL.ImageOps.invert(padded_image)
    
    # Convert image to numpy array and normalize pixel values
    image_array = np.array(list(inverted_image.tobytes())).reshape((1, 28, 28, 1)) / 255.0
    
    # Making POST request with payload_scoring as JSON data
    # Serialize image_array to JSON
    input_data = {"values": image_array.tolist()}
  
    payload_scoring = {"input_data": [input_data]}        

    # Get the token
    # https://cpd-cp4d.data-monetization-7978dc60e1e695936f236140cb8874c5-0000.eu-de.containers.appdomain.cloud/icp4d-api/v1/authorize
    AUTHORIZE_URL = os.getenv('AUTHORIZE_URL')
    WML_USER = os.getenv('WML_USER')
    WML_USER_PASSWORD = os.getenv('WML_USER_PASSWORD')

    form = {"username": WML_USER, "password": WML_USER_PASSWORD}
    data_to_send = json.dumps(form).encode("utf-8")

    token_response = requests.post(AUTHORIZE_URL, data=data_to_send, headers={'Content-Type': 'application/json'})
    print(token_response.json())
    mltoken = token_response.json()["token"]
    print(mltoken)


    # Endpoint URL for making predictions
    # 
    MODEL_URL = os.getenv('MODEL_URL')
    url = MODEL_URL

    # Authorization header with mltoken
    headers = {
        'Authorization': 'Bearer ' + mltoken,
        'Content-Type': 'application/json'  # Specify JSON content type
    }

    response_scoring = requests.post(url, json=payload_scoring, headers=headers)
            
    prediction = response_scoring.json()["predictions"][0]["values"][0]  # Assuming single prediction
    log("HOSTED model")
    predicted_image = str(np.argmax(prediction))
    response = predicted_image  # Convert prediction to string
    log("Predicted Image is : " + response)

    # Check if the request was successful
    if response_scoring.status_code == 200:
        return flask.Response(response, mimetype='application/json', status=200)
    else:
        print("Failed to get response from server, status code:", response.status_code)
        # Return an error response
        return flask.Response("Error calling the model", status=response.status_code)
 
# Get the PORT from environment
port = os.getenv('PORT', '8080')
debug = os.getenv('DEBUG', 'false')
if __name__ == "__main__":
    log("application ready - Debug is " + str(debug))
    app.run(host='0.0.0.0', port=int(port), debug=debug)
