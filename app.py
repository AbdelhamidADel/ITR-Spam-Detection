import numpy as np
from flask import Flask, request, render_template
from project import model_spam 

# pred=model_spam("offer")
# if pred==1:
#     print("Spam Message ")
# elif pred==0:
#     print("Not Spam Message")
app = Flask(__name__,template_folder='templates')
@app.route('/')
@app.route('/Home.html')
def home():
    return render_template('Home.html')

@app.route('/Project',methods=['GET', 'POST'])
@app.route('/Project.html',methods=['GET', 'POST'])
def Project():
    search = request.form.get("text", False)
    pred=model_spam(search)
    if pred[0]==1:
        return render_template('Project.html',output="It's Spam ❎")
    else :
        return render_template('Project.html',output="It's not Spam ✅")
        
      

@app.route('/About.html')
@app.route('/About')
def About():
    return render_template('About.html')

@app.route('/Contact.html')
@app.route('/Contact')
def Contact():
    return render_template('Contact.html')


if __name__ == "__main__":
    app.run(debug=True)