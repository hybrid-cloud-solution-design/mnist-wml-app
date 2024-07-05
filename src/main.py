 #!/usr/bin/env python3

import flask
import os
import requests
import PIL
import PIL.ImageOps
import numpy as np


app = flask.Flask(__name__)

predict_api_url = os.getenv('PREDICT_API_URL', 'http://localhost:8081/image')

def log(e):
    print("{0}\n".format(e))

@app.route('/', methods=['GET'])
def index():
    return flask.render_template("mnist.html")

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
    mltoken = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IkJMNGo3YlY0emVWMXBUak5fZ1RwdzlJVzRWc3FTc0FRTF9vUTBlSTBPLW8ifQ.eyJ1c2VybmFtZSI6ImNwYWRtaW4iLCJyb2xlIjoiQWRtaW4iLCJwZXJtaXNzaW9ucyI6WyJhdXRob3JfZ292ZXJuYW5jZV9hcnRpZmFjdHMiLCJtYW5hZ2VfZ292ZXJuYW5jZV93b3JrZmxvdyIsInZpZXdfZ292ZXJuYW5jZV9hcnRpZmFjdHMiLCJtYW5hZ2VfY2F0ZWdvcmllcyIsIm1hbmFnZV9kaXNjb3ZlcnkiLCJtYW5hZ2VfZGF0YV9xdWFsaXR5X3NsYV9ydWxlcyIsImRhdGFfcXVhbGl0eV9kcmlsbF9kb3duIiwibWFuYWdlX2dsb3NzYXJ5IiwibWFuYWdlX2NhdGFsb2ciLCJhZG1pbmlzdHJhdG9yIiwiY2FuX3Byb3Zpc2lvbiIsIm1vbml0b3JfcGxhdGZvcm0iLCJjb25maWd1cmVfcGxhdGZvcm0iLCJ2aWV3X3BsYXRmb3JtX2hlYWx0aCIsImNvbmZpZ3VyZV9hdXRoIiwibWFuYWdlX3VzZXJzIiwibWFuYWdlX2dyb3VwcyIsIm1hbmFnZV9zZXJ2aWNlX2luc3RhbmNlcyIsIm1hbmFnZV92YXVsdHNfYW5kX3NlY3JldHMiLCJzaGFyZV9zZWNyZXRzIiwiYWRkX3ZhdWx0cyIsImNyZWF0ZV9wcm9qZWN0IiwiY3JlYXRlX3NwYWNlIiwiYWNjZXNzX2NhdGFsb2ciLCJtYW5hZ2VfcmVwb3J0aW5nIiwibW9uaXRvcl9wcm9qZWN0Iiwic2lnbl9pbl9vbmx5Il0sImdyb3VwcyI6WzEwMDAwXSwic3ViIjoiY3BhZG1pbiIsImlzcyI6IktOT1hTU08iLCJhdWQiOiJEU1giLCJ1aWQiOiIxMDAwMzMxMDAxIiwiYXV0aGVudGljYXRvciI6ImV4dGVybmFsIiwiaWFtIjp7ImFjY2Vzc1Rva2VuIjoiYjY1MjIyZmEyMWQyMjY2ZGRiOWUzOWI3NGM5ZDQ0ZGQxNTA5ZTVmYWMyYjllODk0MmVhOTkyOTA3MTQwYWEwMDRmMGIzYmM4MDQ0NWYzZTNkY2Q0YjhhMGQzZjM2MTIxZjkyZDBiNTZhNWYwMDE3MjhkMjNmYWNiYmM4ODc2ZTk2Y2QyYWU5YzcyM2VhMzJhNDkxMTVhNTEyZDE4MDI3NWFhZjJkZDdkZTVhNTM1ZTk5OTFhY2MyZTdjNzZiZjY2MzBmYzQ3MWQ3YmYzMWEyNDI0Mzk5M2Y0OThkOWY5ZDVmZWViYmU2NzBkYjYwYTQ3MjI0MDI1NDRmNDY0ZmUxOGYzODUwYWM2NWJhNTQ2NDVhYzdjYWE2YmZjZDI2ZGM1Y2RmZDgyMzI1ZjBjY2ZkOTQ5NTIzYzY3NGRjNmYyZmZiYTc3MjQyMDI5YTllYTRlM2RjOWM3NGQxMTI3ZTJkYmM0M2M4M2ExZmNhNmQyMGFmMzVhYzRmNmRhNGYzZWY5YzBiMGNhZmViMjliNWE0MGZhNDAzYmIxMTg3YTNlOGYzYWVjYzk3NjkyN2U4NTg1NTc2YjU3ZTUwNzNlYzg4YjEwMjMwNDFiMDI2NjQ5NTAyZDRkOWZjMzRkOWQ4ZDA3M2M2YjEwYWYyN2I4ZDkzOWUzZGQ4OGZiMGU2MDgzMTJmOTNiNjk3ZmJhZmFjZTNiMDljMmFmMTM1MTJiYjI3NGJhYjk0ODc3M2M2MjFjMWI4YjYwZjNiMTVhYmQ5YTA0YjcwYzc2MTFiMTIzYzgzYzRjYmVjZTdlYjJkYTkwMGQ1YTI4ZGFlNGU4OWQ1NjI3NjRkMGQ0ZjFmYTQ0NWViYTU5NjI1ZjI5NzAxOTAxZmRjZGY4NWJjMjAxNzdiMjA5NTllNmI2NDgzZTQ1NzQwMjVhMGFiYTUzZThiODg0NTU4MGM3NTZmMDg0Yzc1YmM1ZTdhOTVjZTQ4Y2FiMjM0YzVkOWQ3MTVlYjdlMDQ0NDNlOWU5MGYyOTQ4ZDc5ZmI0NjRiNTYxNGFmM2Q1OGM5MzIzNWE2ODBhOTAzMjhmNmZjYjk1NThjNjk3ZWFhZTA1MmVjMmM3MTcwMzkwMGMzNDYxYzkwNWIwMTk0YWIzNzI2N2Q1ZWQwZmZkN2U2NzAzMjkxZTk1NmZhZTAzNmVjMjdiMTNjOTM4NDc0YTgxOTUyNzNkNzg3MDA4MTA4YTk0MTVjMTBmM2QyMzRiYmVmZjk2OWYyMGQ0OGQ2YTQ0ZTE4YzMzMjU5OTExZGNhMGYyMzBlMjhiN2UwYzBlMjYzMWI4MmRhZmNmNzUxNzRkMjBhMmYwMTQyYzVjYjY0MDRhNjMxNzMyYjY5ZmRiNDFjNSJ9LCJkaXNwbGF5X25hbWUiOiJjcGFkbWluIiwiYXBpX3JlcXVlc3QiOmZhbHNlLCJpYXQiOjE3MjAxNzc0OTEsImV4cCI6MTcyMDIyMDY5MX0.xkS6yvXnCRzwEF9WcIj1shFqTrMeryByPv_bJBKLZlta_B4XHIaBr65ulhixE0PU-a_-nLcoluIDL8nEEmS9ZKk2y454CWdViQwI5sIIK0liz0XCmU3eJHH535rFFoCmEIqVZ0S8dTVUCIoNLDhDRHSQuUCB69PzMyo5l1T6WihCZGMbSZfMgYQ4fgt3eVZUEqHQFvRM1saaSWmWh0y-pb3YJC2vQAeDm0qZ-ReLo3TmMP9M1rHryPvtMk-bqOrBx0sysbgc_zigGF9g8Dc70h-rKxkuBHBpjqAZKCwKbhyZUWIf4i8lYEjDz8-8xBg6ZpO8xc9NbJkmFqZCabNa4w"
    # Endpoint URL for making predictions
    url = 'https://cpd-cp4d.data-monetization-7978dc60e1e695936f236140cb8874c5-0000.eu-de.containers.appdomain.cloud/ml/v4/deployments/3a530846-65e7-4578-8db3-b8f737953e85/predictions?version=2021-05-01'

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
