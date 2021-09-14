# Лабораторная работа №7

Анализ тональности текстов (RNN)
----------------------------------------------------------------

Using python3.8, develop a neural network based on a recurrent layer to determine the sentiment of sentences of text based on a dataset with an already existing sentiment.

| 🔢  | Ход работы   | ℹ️ |
| ------------- | ------------- |------------- |
| 1️⃣ | Установить библиотеки nltk, gensim.| ✅ |
| 2️⃣ | Скачать dataset c размеченным текстом (позитивный / негативный). |✅  |
| 3️⃣ | Обработать тексты стеммеров, удалить частые слова (stop-words).|✅  |
| 4️⃣ |	Скачать предобученную модель Word2Vec (или самому обучить), с ее помощь получить векторное представление слов.|✅  |
| 5️⃣ | Создать архитектуру нейронной сети с Embedding слоем, рекуррентным слоем (LSTM).|✅  |
| 6️⃣ | Обучить модель, оценить полноту, точность и аккуратность с помощью classification report.|✅  |
| 7️⃣ | Построить график потерь и точности для каждой модели.|✅  |
| 8️⃣ | Выбрать несколько предложений, определить их класс (позитивный или негативный) и визуализировать с помощью TSNE модели.|✅  |



Цель работы
------------
С помощью python3.8 разработать нейронную сеть на основе рекуррентного слоя для определения тональности предложений текста на основе dataset с уже имеющейся тональностью.


Выполнение работы
-----------------

Рекуррентные нейронные сети (RNN) - это класс нейронных сетей, который эффективен
для моделирования данных последовательности, таких как временные ряды или
естественный язык. Достоинство RNN в том, что они обладают памятью за счет наличия
обратных связей «активации» циркулирующих в сети.
В рекуррентных нейронных сетях нейроны обмениваются информацией между собой:
например, вдобавок к новому кусочку входящих данных нейрон также получает
некоторую информацию о предыдущем состоянии сети. Таким образом в сети реализуется
«память», что принципиально меняет характер ее работы и позволяет анализировать
любые последовательности данных, в которых важно, в каком порядке идут значения —
от звукозаписей до котировок акций.

<p align="center">
  <img src="https://static.wixstatic.com/media/3eee0b_969c1d3e8d7943f0bd693d6151199f69~mv2.gif" />
</p>



Таким образом, RNN хорошо подходят для анализа текстов и аудио, где помимо
содержания важен также порядок следования слов.

#### Описание программы

Для лабораторной работы нам нужно создать нейронную сеть и обучить её на основе предоставленных  текстов, чтобы она смогла определить позитивные или негативные предложения. А также самому положить несколько своих предложений, и посмотреть работу нейронной сети.

<p align="center">
  <img src="https://hsto.org/webt/2u/l3/lw/2ul3lwsbyobovjnol2g_cbvrghi.gif" />
</p>

Помимо рекуррентного слоя ещё используется embedding слой, который приводит вектора разных значений к вектору одного размера, то есть приводит к одной форме входных данных. Фрагмент кода с созданием структуры:

```python
model = Sequential()
model.add(Embedding(vocab_size, glove_vectors.vector_size, weights=[embedding_matrix], input_length=X.shape[1]))
model.add(LSTM(25, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

Рассмотрим основные шаги реализации программы:

1. У нас есть массив состоящий из текстов, который в дальнейшем мы редактируем с помощью стеммера и стоп-слов.
Стеммер - это процесс нахождения основы слова для заданного исходного слова. Основа слова не обязательно совпадает с морфологическим корнем слова. К примеру есть слово "привет" и стеммер отрезает все лишнее и оставляет основу "прив", чтобы другие слова схожие по написанию, как "привет", "приветствие", "приветствую" имели одну и туже основу "прив".
Стоп-слова это слова которые не несут в себе никакой смысловой нагрузки. Примерами стоп-слов являются предлоги.
Изначально мы скачиваем/создаем массив стоп-слов. При запуске программа пробегает каждое предложение, проверяет каждое слово и если это слово находится в массиве стоп-слов, то тогда проверяемое слово не записывается в новый текст. А если проверяемое слово не содержится в стоп-словах, то с помощью стеммера отрезаем от него основу и формируем новый текст, состоящий из подобных основ других слов. Фрагмент кода представлен ниже:

```python
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
```

2. После обработки текста, мы используем модель Word2vec она преобразует слова в векторные значений. В качестве Word2vec модели возьмем готовую предобученную модель twitter-100.

3. Так же есть модель tokenizer - это по сути словарь. Она работает следующим образом: по сути у нас есть тексты, которые подаются в эту модель. Она вытаскивает все уникальные слова из текстов, к примеру почти во всех текстах есть одно и то же слово "This", тогда модель берет это слово, вставляет в свой словарь и каждому этому слову дает идентификатор , а этот идентификатор является  числом типа integer, id слов записываются в словаре так  [0, 1, ....]. Отсюда понятно, что модель пробегает все тексты запоминает все уникальные слова, и  модель формирует что-то в виде словаря где Id = слово.  

4. Далее кодируем массив состоящий, тест находящийся в нем  преобразуем в id слов, ранее мы создали словарь у каждого слова там  есть идентификатор, в итоге из каждого предложения вместо слов получили id. В итоге у нас получился зашифрованный тест. Используем метод Pad_sequences, для приведения всех предложений к одинаковой длине с самым длинным предложением. К тем  предложениям которые короче самого длинного, добавляем нули, чтобы выровнять их по длине. Фрагмент кода представлен ниже:

```python
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
```

5. Возвращаясь к Word2vec модели, в эту модель кидаем все слова из словаря, включая уникальные слова. В этой модели для каждого слова даётся вектор в координатах. То есть эта модель из слова делает вектор значений для обработки данных. Запихиваем все эти значения в embedding matrix, и у нас получается матрица в которой содержится для каждого слова из словаря вектор значений.


Далее идёт обучение и оценка нейронной сети, подобное было у нас в предыдущих лабораторных.

# Результат выполнения программы

<p align="center">
  <img src="https://bmstu.codes/MorozoFF/lr-7-opc/-/raw/master/Evaluations.png" />
</p>



 Визуализация с помощью TSNE модели
 -----------------------------------

 <p align="center">
   <img src="https://bmstu.codes/MorozoFF/lr-7-opc/-/raw/master/TSNE_LR7.png" />
 </p>



 График потерь и точности для каждой модели
 ------------------------------------------

Для 4-х эпох

<p align="center">
  <img src="https://bmstu.codes/MorozoFF/lr-7-opc/-/raw/master/loss-accuracy-rnn.png" />
</p>


Для эпох 10 (при 4-х потери увеличиваются)

<p align="center">
  <img src="https://bmstu.codes/MorozoFF/lr-7-opc/-/raw/master/loss-accuracy-rnn__epochs___10__.png" />
</p>

Дополнительный материал
-----------------------

Так же напоминаю для тех кому интересно выполнить задание самому или протестировать данную программу, то прошу перейти [сюда](https://drive.google.com/drive/folders/1b_molbj8z6JhHV6r178AeI1XpQezehsm?usp=sharing "Практикум по машинному обучению")
