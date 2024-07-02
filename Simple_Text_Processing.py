import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download('gutenberg')
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd
os.system("python -m spacy download en_core_web_sm")
from collections import Counter
import csv
from nltk.tag import pos_tag_sents
import en_core_web_sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def stats_pos(csv_file_path):
    """Return the normalized frequency of all appeared part of speech in the questions and answers
    (namely the `sentence text` column) in the given csv file, respectively. Each of the resulting 
    two lists must be sorted alphabetically according to tags.
    Example:
    >>> stats_pos('dev_test.csv')
    output would look like [(ADV, 0.1), (NOUN, 0.21), ...], [(ADJ, 0.08), (ADV, 0.22), ...]
    """ 
    #initializing lists for unique questions and answers
    unique_questions = []
    answers = []
    #reading the CSV file and preparing the data
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question = row['question']
            if question not in unique_questions:
                unique_questions.append(question)
            answers.append(row['sentence text'])

    #concatenating texts
    questions_text = ' '.join(unique_questions)
    answers_text = ' '.join(answers)

    #tokenizing sentences into words
    questions_sentences = sent_tokenize(questions_text)
    answers_sentences = sent_tokenize(answers_text)

    #tokenizing words and tagging part of speech
    questions_tokens = [word_tokenize(sentence) for sentence in questions_sentences]
    answers_tokens = [word_tokenize(sentence) for sentence in answers_sentences]
    questions_pos_tagged = pos_tag_sents(questions_tokens, tagset='universal')
    answers_pos_tagged = pos_tag_sents(answers_tokens, tagset='universal')

    #flattening the tagged lists and counting occurrences of each part of speech
    questions_pos_flat = [tag for sent in questions_pos_tagged for tag in sent]
    answers_pos_flat = [tag for sent in answers_pos_tagged for tag in sent]
    questions_pos_counts = Counter(tag[1] for tag in questions_pos_flat)
    answers_pos_counts = Counter(tag[1] for tag in answers_pos_flat)

    #calculating frequencies and rounding to 4 precision after the decimal point
    total_questions = sum(questions_pos_counts.values())
    total_answers = sum(answers_pos_counts.values())
    questions_pos_freq = [(pos, round(count / total_questions, 4)) for pos, count in sorted(questions_pos_counts.items())]
    answers_pos_freq = [(pos, round(count / total_answers, 4)) for pos, count in sorted(answers_pos_counts.items())]

    return questions_pos_freq, answers_pos_freq

# The parts of speech (PoS) distributions in questions and answers indicates both similarities and differences
# Both questions and answers have high frequencies of nouns, indicating they are used frequently
# ADJ have a higher frequency in answers compared to questions, ADV are more frequent in answers,
# CONJ show a higher frequency in answers, PRON have a higher frequency in questions compared to answers,
# VERB are more frequent in questions than in answers

 

def stats_top_stem_ngrams(csv_file_path, n, N):
    """Return the N most frequent n-gram of stems together with their normalized frequency 
    for questions and answers, respectively. Each is sorted in descending order of frequency
    Example:
    >>> stats_top_stem_ngrams('dev_test.csv', 2, 5)
    output would look like [('what', 'is', 0.43), ('how', 'many', 0.39), ....], [('I', 'feel', 0.64), ('pain', 'in', 0.32), ...]
    """
    #initializing Porter stemmer
    stemmer = nltk.PorterStemmer()
    
    #initializing lists for unique questions and answers
    unique_questions = []
    answers = []

    #reading the CSV file and preparing the data
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question = row['question']
            if question not in unique_questions:
                unique_questions.append(question)
            answers.append(row['sentence text'])

    #concatenating texts
    questions_text = ' '.join(unique_questions)
    answers_text = ' '.join(answers)

    #tokenizing sentences into words
    questions_sentences = sent_tokenize(questions_text)
    sentence_sentences = sent_tokenize(answers_text)
    
    #initializing lists to store n-grams for questions and sentence text
    questions_ngrams = []
    sentence_ngrams = []
    
    #generating n-grams of stems for questions
    for sentence in questions_sentences:
        words = word_tokenize(sentence)
        stems = [stemmer.stem(word.lower()) for word in words]
        ngrams = [tuple(stems[i:i + n]) for i in range(len(stems) - n + 1)]
        questions_ngrams.extend(ngrams)
    
    #generating n-grams of stems for sentence text
    for sentence in sentence_sentences:
        words = word_tokenize(sentence)
        stems = [stemmer.stem(word.lower()) for word in words]
        ngrams = [tuple(stems[i:i + n]) for i in range(len(stems) - n + 1)]
        sentence_ngrams.extend(ngrams)
    
    #counting occurrences of each n-gram for questions and sentence text
    questions_ngram_counts = Counter(questions_ngrams)
    sentence_ngram_counts = Counter(sentence_ngrams)
    
    #calculating total number of n-grams for questions and sentence text
    total_questions_ngrams = len(questions_ngrams)
    total_sentence_ngrams = len(sentence_ngrams)
    
    #calculating frequencies and normalizing for questions
    top_questions_ngrams = [(ngram, round(count / total_questions_ngrams, 4)) for ngram, count in questions_ngram_counts.most_common(N)]
    
    #calculating frequencies and normalizing for sentence text
    top_sentence_ngrams = [(ngram, round(count / total_sentence_ngrams, 4)) for ngram, count in sentence_ngram_counts.most_common(N)]
    
    return top_questions_ngrams, top_sentence_ngrams

#There are some shared bigrams between questions and answers, both have bigrams like (of, the) and (in, the)
#indicating these phrases are used frequently in both types of text



def stats_ne(csv_file_path):
    """Return the normalized frequency of all named entity types for questions and answers, respectively.
    Each is sorted in descending order of frequency.
    Example:
    >>> stats_ne('dev_test.csv')
    output would look like [(DATE, 0.34), ....]
    """
    #loading spaCy English model
    nlp = en_core_web_sm.load()

    #initializing lists for unique questions and answers
    unique_questions = []
    answers = []

    #reading the CSV file and preparing the data
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question = row['question']
            if question not in unique_questions:
                unique_questions.append(question)
            answers.append(row['sentence text'])

    #concatenating texts
    questions_text = ' '.join(unique_questions)
    answers_text = ' '.join(answers)

    #processing texts with spaCy
    question_doc = nlp(questions_text)
    answer_doc = nlp(answers_text[:len(questions_text)])

    #extracting named entities and counting them
    question_entities = [ent.label_ for ent in question_doc.ents]
    answer_entities = [ent.label_ for ent in answer_doc.ents]

    #counting and normalizing
    question_entity_counts = Counter(question_entities)
    answer_entity_counts = Counter(answer_entities)

    total_question_entities = sum(question_entity_counts.values())
    total_answer_entities = sum(answer_entity_counts.values())

    normalized_question_entities = [(ent, round((count / total_question_entities), 4)) for ent, count in sorted(question_entity_counts.items())]
    normalized_answer_entities = [(ent, round((count / total_answer_entities), 4)) for ent, count in sorted(answer_entity_counts.items())]

    return normalized_question_entities, normalized_answer_entities

def stats_tfidf(csv_file_path):
    """Return the ratio of questions that its most similar sentence falls in its answers.
    Example:
    >>> stats_tfidf('dev_test.csv')
    output would be a decimal like 0.38
    """
    #loading dataset
    data = pd.read_csv(csv_file_path)

    #vectorizationing 
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    #combining text columns for a unified corpus
    corpus = data['sentence text'].tolist() 
    corpus = corpus + data['question'].drop_duplicates().tolist()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    #separating the question vectors for focused comparison
    q_vectors = tfidf_matrix[len(data['sentence text']):] 
    s_vectors = tfidf_matrix[:len(data['sentence text'])]

    #iterating through questions, finding similarities directly
    correct_count = 0
    for i, (question, question_vector) in enumerate(zip(data['question'].drop_duplicates(), q_vectors)):
        similarities = cosine_similarity(question_vector, s_vectors)[0]
        most_similar = similarities.argmax()

        if data.iloc[most_similar]['label'] == 1 and data.iloc[most_similar]['question'] == question:
            correct_count += 1

    ratio_correct = round(correct_count / len(data['question'].drop_duplicates()), 4)
    return ratio_correct



# DO NOT MODIFY THE CODE BELOW
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    # print("---------Task 1---------------")
    # print(stats_pos('data/dev_test.csv'))
  
    # print("---------Task 2---------------")
    # print(stats_top_stem_ngrams('data/dev_test.csv', 2, 5))

    # print("---------Task 3---------------")
    # print(stats_ne('data/dev_test.csv'))

    # print("---------Task 4---------------")
    # print(stats_tfidf('data/dev_test.csv'))
  
