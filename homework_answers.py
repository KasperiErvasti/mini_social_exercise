import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import sqlite3


def calculate_optimal_lda(corpus, dictionary, bow_list, upper_range=20):
    optimal_coherence = -100
    optimal_lda = None
    optimal_k = 0
    for K in range(
        2, upper_range
    ):  # result is that 14 is the optimal_k, if upper_range=20
        lda = LdaModel(
            corpus, num_topics=K, id2word=dictionary, passes=10, random_state=2
        )

        coherence_model = CoherenceModel(
            model=lda, texts=bow_list, dictionary=dictionary, coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()

        if coherence_score > optimal_coherence:
            print(
                f'Trained LDA with {K} topics. Average topic coherence (higher is better): {coherence_score} which is the best so far!'
            )
            optimal_coherence = coherence_score
            optimal_lda = lda
            optimal_k = K
        else:
            print(
                f'Trained LDA with {K} topics. Average topic coherence (higher is better): {coherence_score} which is not very good.'
            )

    return optimal_lda, optimal_k


def interpret_sentiment_score(overall_score):
    """Interpreted according to:
    https://hex.tech/templates/sentiment-analysis/vader-sentiment-analysis/
    """
    sentiment = "neutral"
    if overall_score >= 0.05:
        sentiment = "positive"
    elif overall_score <= -0.05:
        sentiment = "negative"

    return sentiment


def main():
    # Exercise 4.1
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    conn = sqlite3.connect("database.sqlite")
    stmt = "SELECT id, content FROM posts"
    data = pd.read_sql_query(stmt, conn)
    conn.close()

    stop_words = stopwords.words('english')

    stop_words.extend(
        [
            'would',
            'best',
            'always',
            'amazing',
            'bought',
            'quick' 'people',
            'new',
            'fun',
            'think',
            'know',
            'believe',
            'many',
            'thing',
            'need',
            'small',
            'even',
            'make',
            'love',
            'mean',
            'fact',
            'question',
            'time',
            'reason',
            'also',
            'could',
            'true',
            'well',
            'life',
            'said',
            'year',
            'going',
            'good',
            'really',
            'much',
            'want',
            'back',
            'look',
            'article',
            'host',
            'university',
            'reply',
            'thanks',
            'mail',
            'post',
            'please',
        ]
    )

    lemmatizer = WordNetLemmatizer()

    bow_list = []

    for _, row in data.iterrows():
        text = row['content']
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if len(t) > 2]
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

        if len(tokens) > 0:
            bow_list.append(tokens)

    dictionary = Dictionary(bow_list)

    dictionary.filter_extremes(no_below=2, no_above=0.3)
    corpus = [dictionary.doc2bow(tokens) for tokens in bow_list]

    # Used this to get the optimal_k
    # optimal_lda, optimal_k = calculate_optimal_lda(corpus, dictionary, bow_list, upper_range=20)

    # optimal k is 14 when tried for 2-19 topics (only tested up to 19 because takes too long)
    optimal_k = 14
    optimal_lda = LdaModel(
        corpus, num_topics=optimal_k, id2word=dictionary, passes=10, random_state=2
    )

    topic_words_lists = {}
    for i, topic in optimal_lda.print_topics(num_words=5):
        topic_words_lists[i] = topic

    topic_counts = [0] * optimal_k
    for bow in corpus:
        topic_dist = optimal_lda.get_document_topics(bow)
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
        topic_counts[dominant_topic] += 1

    topic_list = [(i, count) for i, count in enumerate(topic_counts)]
    topic_list.sort(key=lambda x: x[1], reverse=True)

    for topic_num, count in topic_list[:10]:
        print(
            f"Topic {topic_num+1}: {count} posts, represented by words: {topic_words_lists[topic_num]}"
        )

    # Exercise 4.2
    print("------------------------")
    print("Ex 4.2")
    nltk.download('vader_lexicon')

    conn = sqlite3.connect("database.sqlite")
    stmt = "SELECT id, content FROM posts"
    post_data = pd.read_sql_query(stmt, conn)
    stmt = "SELECT id, content FROM comments"
    comment_data = pd.read_sql_query(stmt, conn)
    conn.close()
    combined_df = pd.concat([post_data, comment_data], ignore_index=True)

    sia = SentimentIntensityAnalyzer()
    combined_df['sentiment_score'] = combined_df['content'].apply(
        lambda content: sia.polarity_scores(content)['compound']
    )

    overall_sentiment = combined_df['sentiment_score'].mean()
    print(f"Overall sentiment score (across post and comments): {overall_sentiment:.3f}")
    sentiment = interpret_sentiment_score(overall_sentiment)
    print(f"Overall sentiment is {sentiment}")

    # topic calcs
    topic_sentiment_scores = []

    for topic_num, _ in topic_list[:10]:
        topic_posts = []
        for i, bow in enumerate(corpus):
            topic_dist = optimal_lda.get_document_topics(bow)
            dominant_topic = max(topic_dist, key=lambda item: item[1])[0]
            if dominant_topic == topic_num:
                topic_posts.append(post_data.iloc[i]['content'])

        if topic_posts:
            scores = [sia.polarity_scores(text)['compound'] for text in topic_posts]
            avg_score = sum(scores) / len(scores)
            topic_sentiment_scores.append((topic_num, avg_score))

    print("\nTop 10 topic sentiments")
    for topic_num, avg_score in topic_sentiment_scores:
        sentiment = interpret_sentiment_score(avg_score)
        print(f"Topic {topic_num+1}: {avg_score:.2f} ({sentiment})")


if __name__ == '__main__':
    main()
