# Лабораторная работа №7

Анализ тональности текстов (RNN)
----------------------------------------------------------------



| 🔢  | Ход работы   | ℹ️ |
| ------------- | ------------- |------------- |
| 1️⃣ | Установить библиотеки nltk, gensim.| ✅ |
| 2️⃣ | Скачать датасет c размеченным текстом (позитивный / негативный). |✅  |
| 3️⃣ | Обработать тексты стеммеров, удалить частые слова (stopwords).|✅  |
| 4️⃣ |	Скачать предобученную модель Word2Vec (или самому обучить), с ее помощь получить векторное представление слов.|✅  |
| 5️⃣ | Создать архитектуру нейронной сети с Embedding слоем, рекуррентным слоем (LSTM) |✅  |
| 6️⃣ | Обучить модель, оценить полноту, точность и аккуратность с помощью classification report.|✅  |
| 7️⃣ | Построить график потерь и точности для каждой модели.|✅  |
| 8️⃣ | Выбрать несколько предложений, определить их класс (позитивный или негативный) и визуализировать с помощью TSNE модели.|✅  |



Цель работы
------------
С помощью python3.8 разработать нейронную сеть на основе рекурентного слоя для определения тональности предложений текста на основе dataset с уже имеющейся тональностью.


Выполнение работы
-----------------
Нам нужно создать нейросеть и обучить её на основе текстов, чтобы она смогла определить позитивные и негативные предложения. А также самому положить несколько своих предложений, и посмотреть работу нейронной сети.
Помимо рекурентного слоя ещё используется embedding слой, который приводит вектора разных значений к вектору одного размера, то есть приводит к одной форме входных данных.


У нас есть массив из текстов, который в дальнейшем мы редактируем с помощью стеммера и стоп-слов.
Стеммер это процесс нахождения основы слова для заданного исходного слова. Основа слова не обязательно совпадает с морфологическим корнем слова. К примеру есть слово "привет" и стеммер отрезает все лишнее и оставляет основу "прив", чтобы другие слова схожие по написанию, как "привет", "приветствие", "приветсвую" имели одну и туже основу "прив".
Стоп-слова это слова которые не несут в себе никакой смысловой нагрузки. Примерами стоп-слов являются предлоги.
Изначально мы скачиваем/создаем массив стоп-слов. При запуске программа пробегает каждое предложение, проверяет каждое слово и если это слово находится в массиве стоп-слов, то тогда проверяемое слово не записывается в новый текст. А если проверяемое слово не содержится в стоп-словах, то с помощью стеммера отрезаем от него основу и формируем новый текст, состоящий из подобных основ других слов.


Мы обработали текст первично, далее мы используем модель V2Vec она преобразует слова в векторные значений. В качестве V2Vec модели возьмем готовую предобученную модель twitter-100.
Так же есть модель токинайзер это по сути словарь. Она работает следующим образом: по сути у нас есть тексты, которые подаются в эту модель. Она вытаскивает все уникальные слова из текстов, к примеру почти во всех текстах есть одно и то же слово "This", тогда модель берет это слово, вставляет в свой словарь и каждому этому слову дает идентификатор , а этот идентификатор является  числом типа integer, id слов записываются в словаре так  [0, 1, ....]. Отсюда понятно, что модель пробегает все тексты запоминает все уникальные слова, и  модель формирует что-то в виде словаря где Id = слово.  
Далее кодируем массив состоящий, тест находящийся в нем  преобразовуем в id слов, ранее мы создали словарь у каждого слова там  есть идентификатор, в итоге из каждого предложения вместо слов получили id. В итоге у нас получился зашифрованый тест. Используем метод Pad_sequences, для приведения всех предложений к одинаковой длинне с самым длинным предложением. К тем  предложениям которые короче самого длинного, добовляем нули, чтобы выравнить их по длине.


Возвращаясь к V2Vec модели, в эту модель кидаем все слова из словоря, включая уникальные слова. В этой модели для каждого слова даётся вектор в координатах. То есть эта модель из слова делает вектор значений для обработки данных. Запихиваем все эти значения в embedding matrix, и у нас получается матрица в которой содержится для каждого слова из словаря вектор значений.
Далее идёт обучение нейронной сети, обучение, тестовая выборка, оценка нейронной сети.  

# Результат выполнения программы


![Gitlab logo](https://bmstu.codes/MorozoFF/lr-7-opc/-/raw/master/Evaluations.png)

![Gitlab logo](https://bmstu.codes/MorozoFF/lr-7-opc/-/raw/master/Evaluations.png)


 Визуализация с помощью TSNE модели
 -----------------------------------

![Gitlab logo](https://bmstu.codes/MorozoFF/lr-7-opc/-/raw/master/TSNE_LR7.png)


 График потерь и точности для каждой модели
 ------------------------------------------

Для 4-х эпох

![Gitlab logo](https://bmstu.codes/MorozoFF/lr-7-opc/-/raw/master/loss-accuracy-rnn.png)

Для эпох 10 (При 4-х потери увеличиваются)


![Gitlab logo](https://bmstu.codes/MorozoFF/lr-7-opc/-/raw/master/loss-accuracy-rnn__epochs___10__.png)
