import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional # Классы для создания слоёв  
import datetime

# Документ с данными для обучения 
with open('data/data.txt', 'r') as f:
    lines = f.readlines()

# Обработка данных
questions = []
answers = []
for line in lines:
    if line.startswith('-'):
        answers.append(line.strip())
    else:
        questions.append(line.strip())

# Токенизация данных / вычисление уникальных слов
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
total_words = len(tokenizer.word_index) + 1

#--------------------- Входные и выходные данные -------------
input_sequences = [] # Хранение входных последовательностей 
for line in questions + answers: # Объединение 
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Входные, выходные данные 
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)
#---------------------------------------------------------
# Создание модели 
model = Sequential()
model.add(Embedding(total_words, 128, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#===============Обучение и сохранение
# epochs - это число эпох, на которое будет обучаться нейронная сеть.
# verbose=1 вывод информации после каждой epochs. verbose=0 - отключить вывод информации.
model.fit(X, y, epochs=100, verbose=1) 
model.save('chatbot_model.h5')
#====================================
# Функция для генерации вопроса
def generate_answer(seed_text):
    next_words = max_sequence_len - 1
    for _ in range(next_words):
        token_list =tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Функция для генерации ответа, включающая текущую дату |beta 
def generate_response(seed_text):
    generated_answer = generate_answer(seed_text)
    if "сегодня" in seed_text.lower():
        current_date = datetime.date.today().strftime("%d %B %Y")
        generated_answer += " " + current_date
    return generated_answer

# Мини чат
while True:
    question = input()
    print("Ты:", question)
    print("AI:", generate_response(question))
