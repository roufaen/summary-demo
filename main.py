from flask import Flask, render_template, request
from src.infer import Summarizer, initialize, InferConfig

app = Flask('summary', static_folder='statics', static_url_path='/statics')
initialize()
summarizer = Summarizer(InferConfig())


@app.route('/', methods=['GET'])
def get_page():
    return render_template('main.html')


@app.route('/', methods=['POST'])
def get_summary():
    query = request.form['query']
    summary = summarizer.get_summary([query])
    return render_template('main.html', query=query, summary=summary)


if __name__ == '__main__':
    app.run('0.0.0.0', '8080')
