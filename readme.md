# Steps to Run the Project

1. **Train the Model**
   Run the `train_model.py` script to train the machine learning model and save it as `model.pkl`:
   ```bash
   python train_model.py
   ```

2. **Build the Docker Image**
   Build the Docker image for the Flask application:
   ```bash
   docker build -t ml-docker-app .
   ```

3. **Run the Docker Container**
   Start the Docker container:
   ```bash
   docker run -p 5000:5000 ml-docker-app
   ```

4. **Test the API**
   Use a tool like `curl` or Postman to send a POST request to the `/predict` endpoint:
   ```bash
   curl -X POST -H "Content-Type: application/json" \
   -d '{"features": [5.1, 3.5, 1.4, 0.2]}' \
   http://localhost:5000/predict
   ```

   Expected Response:
   ```json
   {
       "prediction": "setosa"
   }