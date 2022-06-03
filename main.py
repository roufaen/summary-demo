from flask import Flask, render_template, request
from src.infer import initialize, Segmentator, SegmentConfig, DyleInfer
import json

app = Flask('summary', static_folder='statics', static_url_path='/statics')
summarizer = None
segmentator = None


@app.route('/', methods=['GET'])
def get_page():
    return render_template('main.html')


@app.route('/', methods=['POST'])
def get_summary():
    query = request.form['query']
    segs = segmentator.get_segments(query)
    summary = summarizer.get_summary(segs)
    # return render_template('main.html', query=query, summary=summary
    # print(summary)
    return json.dumps(summary, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    initialize()
    summarizer = DyleInfer()
    segmentator = Segmentator(SegmentConfig())
    app.run('0.0.0.0', '8080')
