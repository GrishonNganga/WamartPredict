import pandas as pd
import numpy as np
from flask import Flask, jsonify, make_response, request, abort, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("savmodel.pkl", "rb"))

@app.route("/")
def indexpage():
  return render_template('index.html')

@app.route("/get_prediction", methods=['POST','OPTIONS'])
#@cross_origin()
def get_prediction():
  if "Store" in request.form and "Dept" in request.form and "IsHoliday" in request.form:
    store = request.form["Store"]
    dept = request.form["Dept"]
    isHoliday = int(request.form["IsHoliday"])
    isHoliday = 1 if isHoliday == 1 else 0
    to_predict = [[store, dept, isHoliday]]
    to_predict = np.array(to_predict)
    print(to_predict)
 
    ans = model.predict(to_predict)
    print(type(ans))
    return render_template("index.html", ans = ans[0])

# Connecting to other pages 
@app.route('/about')
def about():
	return render_template('about.html', title = "About Us")

@app.route('/contact')
def contact():
	return render_template('contact.html', title = "Contact Us")


if __name__ == "__main__":
  app.run(debug=True)



  # Helpful links
#https://soshace.com/how-i-built-an-admin-dashboard-with-python-flask/
#https://betterprogramming.pub/building-your-first-website-with-flask-part-2-6324721be2ae
#https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/
#https://pythonhow.com/adding-more-pages-to-the-website/