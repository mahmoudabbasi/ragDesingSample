from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = Flask(__name__)
app.config['MONGO_URI'] = 'mongodb://192.168.53.57:27017/booksdb'  
mongo = PyMongo(app)

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def create_sample_data():
    if mongo.db.books.count_documents({}) == 0:
        sample_books = [
            {"title": "جنگ و صلح", "author": "لئو تولستوی", "year": 1869},
            {"title": "1984", "author": "جورج اورول", "year": 1949},
            {"title": "کشتن مرغ مقلد", "author": "هارپر لی", "year": 1960},
            {"title": "مردی در جستجوی معنا", "author": "ویکتور فرانکل", "year": 1946},
            {"title": "غرور و تعصب", "author": "جین آستین", "year": 1813},
            {"title": "صد سال تنهایی", "author": "گابریل گارسیا مارکز", "year": 1967},
            {"title": "بزرگترین فروشنده دنیا", "author": "اوگ ماندینو", "year": 1968},
            {"title": "دنیای سوفی", "author": "یوستین گوردر", "year": 1991},
            {"title": "کیمیاگر", "author": "پائولو کوئیلو", "year": 1988},
            {"title": "مردی که می‌خندد", "author": "ویکتور هوگو", "year": 1869},
            {"title": "بوف کور", "author": "صادق هدایت", "year": 1937},
            {"title": "شازده کوچولو", "author": "آنتوان دو سنت اگزوپری", "year": 1943},
            {"title": "جزیره گنج", "author": "رابرت لویی استیونسن", "year": 1883},
            {"title": "هری پاتر و سنگ جادو", "author": "جی.کی. رولینگ", "year": 1997},
            {"title": "خودآموزی در علم اقتصاد", "author": "توماس سورل", "year": 1996}
        ]
        mongo.db.books.insert_many(sample_books)

@app.route('/')
def index():
    create_sample_data() 
    return "Sample data created!"

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    question_embedding = model.encode(question, convert_to_tensor=True)

    books = mongo.db.books.find()
    book_embeddings = []
    book_titles = []

    for book in books:
        book_title = f"{book['title']} نوشته {book['author']} ({book['year']})"
        book_titles.append(book_title)
        book_embeddings.append(model.encode(book_title, convert_to_tensor=True))


    book_embeddings = np.array(book_embeddings)
    similarities = util.pytorch_cos_sim(question_embedding, book_embeddings)

    closest_index = similarities.argmax()
    closest_book = book_titles[closest_index]

    return jsonify({'answer': closest_book})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
