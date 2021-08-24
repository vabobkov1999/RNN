import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
from sklearn.manifold import TSNE
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import  Dense, Embedding, LSTM
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
from tqdm import tqdm

# Расположение файла с датасетом и название графика, который мы сохраним
train_file_1 = "dataset/amazon_cells_labelled.csv"
train_file_2 = "dataset/imdb_labelled.csv"
train_file_3 = "dataset/yelp_labelled.csv"
plot_path = "loss-accuracy-rnn.png"

# Считываем данные с csv-файлов
data = []
tmp = pd.read_csv(train_file_1)
data.append(tmp)
tmp = pd.read_csv(train_file_2)
data.append(tmp)
tmp = pd.read_csv(train_file_3)
data.append(tmp)
data = pd.concat(data)
print()
print(f"data shape: {data.shape}")
print("[INFO] data read...TRUE...")
print()

# Преобразуем предложенния к виду, где присутствуют только основы слов и удалены стоп-слова (предлоги и т.п.)
nltk.download("stopwords")
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
docs = []
for sentence in tqdm(data["text"]):
    tokens = []
    for token in sentence.split():
        if token not in stop_words:
            tokens.append(stemmer.stem(token))
    docs.append(" ".join(tokens))
print()
print("[INFO] data processing (stemmer and stow words)...TRUE...")
print()

# Создаем словарь с помощью Токинайзера, кодируем числами все слова из словаря, которые содержатся в в предложениях
# Далее с помощью модели V2Vec для каждого слова из словаря получаем его вектор в этом пространстве
# После создаем матрицу, где будут содержаться слова из словаря Токинайзера в векторном представлении модели V2Vec
# В качестве V2Vec модели возьмем готовую предобученную модель twitter-100
import gensim.downloader
glove_vectors = gensim.downloader.load('glove-twitter-100')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)
vocab_size = len(tokenizer.word_index) + 1
print("vocab_size = ", vocab_size)
encoded_docs = tokenizer.texts_to_sequences(docs)
X = pad_sequences(encoded_docs)
Y = data["emotion"]
embedding_matrix = np.zeros((vocab_size, glove_vectors.vector_size))
for word, i in tqdm(tokenizer.word_index.items()):
    if word in glove_vectors:
        embedding_matrix[i] = glove_vectors[word]
print()
print("[INFO] creating embedding matrix...TRUE...")
print()

# разбиваем данные на обучающую и тестовую выборки, используя 80% данных для обучения и оставшиеся 20% для тестирования
(trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.2, random_state=42)
tmp_predictions_0 = testY
labels = Y.unique().tolist()
encoder = LabelEncoder()
encoder.fit(Y.tolist())
trainY = encoder.transform(trainY.tolist())
testY = encoder.transform(testY.tolist())
print("trainX shape:", trainX.shape)
print("trainY shape:", trainY.shape)
print("testX shape:", testX.shape)
print("testY shape:", testY.shape)
print()

# Создаем структуру нейронки
# Эмбединг-ЛСТМ-Денс слои
model = Sequential()
model.add(Embedding(vocab_size, glove_vectors.vector_size, weights=[embedding_matrix], input_length=X.shape[1]))
model.add(LSTM(25, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# обучаем нейросеть
print()
print("[INFO] training network...")
epochs = 4
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs, batch_size=32)

# оцениваем нейросеть
print("[INFO] evaluating network...")
score = model.evaluate(testX, testY)
print()
print("accuracy: ", score[1])
print("loss: ", score[0])
print()

# Предсказываем тональность предложений из 20% выборки, на основе этого строим classification_report
predictions = model.predict(testX, verbose=1, batch_size=32)
tmp_predictions = []
for prediction in predictions:
    if (prediction < 0.5):
        tmp_predictions.append(0)
    else:
        tmp_predictions.append(1)
print(classification_report(tmp_predictions_0, tmp_predictions))

# строим графики потерь и точности
plt.style.use("ggplot")
N = np.arange(1, epochs+1)
plt.figure()
plt.plot(N, H.history["loss"], label="training loss")
plt.plot(N, H.history["val_loss"], label="validation loss")
plt.plot(N, H.history["accuracy"], label="training accuracy")
plt.plot(N, H.history["val_accuracy"], label="validation accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(plot_path)
plt.show()

# Часть кода, где предсказываем принадлежность предложений к классу с помощью обученой нейронки
test_words = []
test_words.append("Call me a nut, but I think this is one of the best movies ever") # 1
test_words.append("What SHOULD have been a hilarious, yummy Christmas Eve dinner to remember was the biggest fail of the entire trip for us.") # 0
test_words.append("useless phone, simply deaf.") # 0
test_words_stem = []
for sentence in test_words:
    tokens = []
    for token in sentence.split():
        if token not in stop_words:
            tokens.append(stemmer.stem(token))
    test_words_stem.append(" ".join(tokens))
encoded_test_words = tokenizer.texts_to_sequences(test_words_stem)
# Создаем TSNE-модель для визуализации слов одного из предложений (их расположение) вместе с словами из словаря
tsne_model = TSNE(n_components=2, perplexity=10, learning_rate=150)
tmp = tsne_model.fit_transform(embedding_matrix)
fig, ax = plt.subplots()    #SAME as: fig = plt.figure()
#                                     ax = fig.add_subplot(111)
ax.set(title='TSNE for LR7')
ax.scatter(tmp[:,0], tmp[:,1], color='green', label='vocabulary', s=2)
dots_for_graph = encoded_test_words[0]
text_for_graph = []
for token in test_words[0].split():
    if token not in stop_words:
        text_for_graph.append(token)
counter = 0
for i in dots_for_graph:
    ax.scatter(tmp[i][0], tmp[i][1], color='red', s=15)
    plt.text(tmp[i][0], tmp[i][1], text_for_graph[counter], fontsize=15)
    counter += 1
ax.legend()
plt.show()
fig.savefig('TSNE_LR7.png')

# предсказываем принадлежность текста к классу и выводим результат в консоль
encoded_test_words = pad_sequences(encoded_test_words, maxlen=len(X[1]))
scores = model.predict(encoded_test_words, verbose=1, batch_size=32)
counter = 0
for i in test_words:
    label = ""
    if scores[counter][0] < 0.5:
        label = "NEGATIVE"
    else:
        label = "POSITIVE"
    print(f"label: {i}", f"score: {scores[counter][0]}", f"predict: {label}")
    counter += 1
