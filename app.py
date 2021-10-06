from flask import Flask, render_template, request, jsonify, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sentiment_analysis')
def sa():
    return render_template('sa.html')

@app.route('/entity_extraction')
def entity_extraction():
    return render_template('entity_extraction.html')

@app.route('/qna')
def qna():
    return render_template('qna.html')

if __name__ == "__main__":
    app.run()