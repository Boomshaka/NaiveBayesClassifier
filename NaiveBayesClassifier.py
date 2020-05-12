from math import log, ceil
import time
import sys


#Run a test on file given as parameter.
#file = filename
#Pos_review and Neg_review = the feature vectors of positive & negatvie reviews
def test(file, pos_review, neg_review):
    start_testing_time = time.time()
    with open(file) as testing:
        data = testing.readlines()
        label = [] #list of labels given by classifier
        compare_label = [] #list of tuples containing classifier label vs actual label
        
        #convert review into list of words, then classify the list of words to compute label
        for document in data:
            words = document.split(' ')[:-1]# Remove number label
            guess = None
            guess_pos = pos_review.classify(words)
            guess_neg = neg_review.classify(words)
            
            guess = 1 if guess_pos > guess_neg else 0
            if document[-2] == '0':
                actual = 0
            else:
                actual = 1
            label.append(guess)
            compare_label.append([guess, actual])

    testing_time = ceil(time.time() - start_testing_time)

    #iterate through compare_label to find out accuracy of NBC by comparing results with answers
    num_correct = 0
    for i in compare_label:
        if i[0] == i[1]:
            num_correct += 1
    accuracy = round(num_correct / len(label), 3)

    return {'label': label, 'accuracy': accuracy, 'testing_time': testing_time}





class Reviews():

    #list of stop words referred from NLTK stop words
    #https://gist.github.com/sebleier/554280
    stop_words = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", 
                "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", 
                "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", 
                "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", 
                "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", 
                "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", 
                "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", 
                "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", 
                "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", 
                "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", 
                "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", 
                "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", 
                "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", 
                "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", 
                "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", 
                "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "ve",
                "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]

    all_words = {}#dict of all words in every single document, and their # of occurence

    def __init__(self, reviews, review_count):
        self.reviews = reviews
        self.frequency = len(reviews) / review_count

        #keeps track of the total # of words in all the documents for a given class (positive or negative)
        self.total_word_count = 0
        self.vector = {}
     


    def create_vector(self):
        for review in self.reviews:
            self.total_word_count += len(review)
            for index, word in enumerate(review):
                if self.vector.get(word) is not None:
                    self.vector[word]['count'] += 1
                    Reviews.all_words[word]['count'] += 1
                else:
                    self.vector.update({word: {'count': 1}})
                    if Reviews.all_words.get(word) is None:
                        Reviews.all_words.update({word: {'count': 1}})
                    else:
                        Reviews.all_words[word]['count'] += 1
            
        a = 20 #Smooth factor, 18~20 seems to be the sweet spot
        for key, value in self.vector.items():
            self.vector[key]['givenC'] = (value['count'] + a) / (self.total_word_count + a * len(self.vector))
        self.smooth = a / (self.total_word_count + a * len(self.vector))


    #Classify each document using Maximum a posteriori with underflow protection
    def classify(self, words):
        prob = 1
        for word in filter(lambda w: w not in Reviews.stop_words, words):
            if self.vector.get(word):
                prob += log(self.vector[word]['givenC'])
            else:
                prob += log(self.smooth)

        return prob + log(self.frequency)






if __name__ == "__main__":
    if (len(sys.argv) != 3):
        error('Incorrect number of arguments')
        exit(1)
    
    training_file = sys.argv[1]
    testing_file = sys.argv[2]

    #number of positive & negative reviews
    num_pos = 0
    num_neg = 0

    #list of all positive & negative reviews
    pos_review_list = []
    neg_review_list = []

    #dicts containing result of testing
    training_test = {}
    testing_test = {}

    start_training_time = time.time()


    with open(training_file) as training:
        data = training.readlines()
        for line in data:
            if(line[-2] == '0'):
                num_neg += 1
                word_list = line.split(' ')[:-1]
                neg_review_list.append(word_list)
            else:
                num_pos += 1
                word_list = line.split(' ')[:-1]
                pos_review_list.append(word_list)


    total_review = num_pos + num_neg
        
    pos_reviews = Reviews(pos_review_list, total_review)
    neg_reviews = Reviews(neg_review_list, total_review)
    
    pos_reviews.create_vector()
    neg_reviews.create_vector()

    # for key, value in pos_reviews.vector.items():
    #     print(f"{key}: {value}")

    # # print(f"Positive vector size: {len(pos_reviews.vector)}")
    # # print(f"Negative vector size: {len(neg_reviews.vector)}")
    # # print(f"ALL words: {len(Reviews.all_words)}")


    training_time = ceil(time.time() - start_training_time)

    testing_test = test(testing_file, pos_reviews, neg_reviews)
    training_test = test(training_file, pos_reviews, neg_reviews)

    for i in testing_test['label']:
        print(i)
    print(f"{training_time} seconds (training)")
    print(f"{testing_test['testing_time']} seconds (labeling)")
    print(f"{training_test['accuracy']} (training)")
    print(f"{testing_test['accuracy']} (testing)")
