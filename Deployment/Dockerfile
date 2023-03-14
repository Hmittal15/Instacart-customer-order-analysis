FROM python:3.11.1

# Pull the base docker image of python with tag 3.11.1

RUN pip install --upgrade pip

WORKDIR /app
# Change the working dir inside the container - cd /app

ADD Derived_data/ /app/Derived_data/

ADD app.py get_prediction.py index.html new_user_recommendation.html predict.html random_forest_model.pkl requirements.txt /app/

RUN pip install -r requirements.txt
# Install the dependency

### Container Env same as local at this point

EXPOSE 8080

CMD ["python", "app.py"]