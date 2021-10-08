from flask import Flask, render_template, request, jsonify, url_for
from modules.qnasingle.main_multiple_ee import loc_main
from modules.extractors.extractor import NRIC_Finder, Phone_Finder

from modules.sentiment.sen import main_sen

from modules.qnasingle.main_multiple import main as main_single
# from qnamultiple.main import main as main_mulitple

app = Flask(__name__)

# Display pages endpts

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

# Process stuff endpts

@app.route('/run-ee-main', methods=['POST'])
def run_ee_main():
    text = request.form.get('text')
    loc = loc_main(text)
    nric = NRIC_Finder(text)
    phone = Phone_Finder(text)

    res = {}
    res['text'] = text
    res['loc'] = loc
    res['nric'] = nric
    res['phone'] = phone
    response = jsonify(res)
    return response

@app.route('/run-sen-main', methods=['POST'])
def run_sen_main():
    text = request.form.get('text')
    model = request.form.get('model')
    sentiment, score = main_sen(text, model)

    res = {}
    res['sen'] = sentiment
    res['score'] = float(score)
    print(res)
    response = jsonify(res)
    return response

@app.route('/run-qna-main-single', methods=['POST'])
def run_qna_main_single():
    text = request.form.get('text')
    qn = request.form.get('qn')
    ans, prob = main_single(text, qn)

    res = {}
    res['ans'] = ans
    res['prob'] = prob
    response = jsonify(res)
    return response

# @app.route('/run-qna-main-multiple', methods=['POST'])
# def run_qnamain():
#     text = request.form.get('text')
#     qn = request.form.get('qn')
#     ans, context, prob = main_mulitple(text, qn)

#     res = {}
#     res['ans'] = ans
#     res['context'] = context
#     res['prob'] = prob

#     response = jsonify(res)
#     return response

if __name__ == "__main__":
    app.run()