#loading libraries
import joblib
from flask import Flask, request, url_for
from get_prediction import get_recommendations

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__,template_folder="C:\\Users\\hmitt\\PycharmProjects\\InstacartModelDeployment\\venv")
# render_template

@app.route('/')
def home():

    """Serve homepage template."""
    #return flask.render_template("C:\\Users\\hmitt\\PycharmProjects\\InstacartModelDeployment\\venv\\index.html")
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    to_predict_list = request.form.to_dict()
    predictions, time = get_recommendations(to_predict_list)
    #print(predictions, time)
    if 'recommend' not in predictions.keys():
        return flask.render_template('new_user_recommendation.html',predictions = predictions)
        #return flask.render_template("C:\\Users\\hmitt\\PycharmProjects\\InstacartModelDeployment\\venv\\new_user_recommendation.html",predictions = predictions)

    #return flask.render_template("C:\\Users\\hmitt\\PycharmProjects\\InstacartModelDeployment\\venv\\predict.html",predictions = predictions)
    return flask.render_template('predict.html',predictions = predictions)

if __name__ == '__main__':
    #app.debug = True
    app.run(host='0.0.0.0', port=8080)