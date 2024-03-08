import numpy as np

class Embeddings:
    
    def __init__(self, file):
        self.embeddings = {}    # embeddings of words
        self.past_guesses = []  # keep track of player's past_guesses
        self.word_rank = {}
        for idx, line in enumerate(open(file)):
            row = line.split()
            word = row[0]
            vals = np.array([float(x) for x in row[1:]])
            self.embeddings[word] = vals
            self.word_rank[word] = idx + 1
        # TODO: Preprocessing
            
    def get_word_embedding(self, word):
        return self.embeddings[word]
    
    def get_associated_word(self, word1, word2):
        # deal with edge cases of one word not being in vocabulary
        if word1 not in self.embeddings:
           return self.most_similar(word2)
        if word2 not in self.embeddings:
            return self.most_similar(word1) 
        
        word1_vector = self.embeddings[word1]
        word2_vector = self.embeddings[word2]

        associated_word = self.most_similar_two(word1_vector, word2_vector)
        return associated_word
    
    def most_similar(self, v1):
        # Return the most similar word to `vector` and its similarity.
        similarities = []

        # going through every word in vocabulary
        # TODO: more efficient method?
        for word in self.embeddings:
            if word not in self.past_guesses:
                word_embedding = self.embeddings[word]
                similarity = (word, cosine_similarity(v1, word_embedding))
                similarities.append(similarity)

        sort_similarities = sorted(similarities, key=lambda x: x[1], reverse = True)

        return sort_similarities[0][0]

    # two number most_similar
    def most_similar_two(self, v1, v2):
        # Return the most similar word to `vector` and its similarity.
        similarities = []

        # going through every word in vocabulary
        # TODO: more efficient method?
        for word in self.embeddings:
            if word not in self.past_guesses:
                word_embedding = self.embeddings[word]
                similarity = (word, self.analogy_similarity(v1, v2, word_embedding))
                similarities.append(similarity)

        sort_similarities = sorted(similarities, key=lambda x: x[1], reverse = True)
        # print top 10 results for debugging
        # print(sort_similarities[:10])

        return sort_similarities[0][0]

    def analogy_similarity(self, v1, v2, target_word_embedding):
        #analogy-based similarity between three vectors. returns an analogy-based similarity float between three words.
        if len(target_word_embedding) < len(v1):
            target_word_embedding = np.append(target_word_embedding, np.zeros((len(v1)- len(target_word_embedding))))

        expected_vector = v1 + v2
        similarity = cosine_similarity(target_word_embedding, expected_vector)
        return similarity
        

# helper functions
def vector_norm(vec):
    return np.sqrt(np.sum(np.square(vec)))

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (vector_norm(v1) * vector_norm(v2))

# telepathy game function
def telepathy_game(player1, player2, player1_word, player2_word, max_iterations= 10):
    converged = False

    iteration = 0

    while not converged and iteration < max_iterations:
        # append first words to past guesses
        player1.past_guesses.append(player1_word)
        player1.past_guesses.append(player2_word)
        player2.past_guesses.append(player2_word)
        player2.past_guesses.append(player1_word)

        # Check if the players have converged on the same word
        # similarity = cosine_similarity(player1.get_word_embedding(player1_word), player2.get_word_embedding(player2_word))
        # print(similarity)

        print("Word Number: ", iteration)
        print("Player 1: " + player1_word)
        print("Player 2: " + player2_word)
        if player1_word == player2_word:  # Must converge on same word to win
            converged = True
            print(f"Players converged on the word: {player1_association}")
            return
        # Player 1 thinks of an associated word
        player1_association = player1.get_associated_word(player1_word, player2_word)

        # Player 2 thinks of an associated word
        player2_association = player2.get_associated_word(player1_word, player2_word)
        # Update the words for the next round
        player1_word = player1_association
        player2_word = player2_association

        iteration += 1

    print("DID NOT CONVERGE.")
if __name__ == '__main__':
    player1_file = '/projects/e31408/data/a5/glove_top50k_50d.txt' # glove ( TODO: look into what this source is )
    player2_file = 'twitter_glove/glove.twitter.27B.50d.txt' # twitter
    # need to be same dimensions
    player1 = Embeddings(file = player1_file)
    player2 = Embeddings(file = player2_file)

    print("Enter a starting word for Player 1.")
    player1_word = str(input()).lower().strip()

    print("Enter a starting word for Player 2.")
    player2_word = str(input()).lower().strip()
    
    # Initialize starting words for both players
    # player1_word = "she"
    # player2_word = "party"

    telepathy_game(player1, player2, player1_word, player2_word)
