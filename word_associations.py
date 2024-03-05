import numpy as np

class Embeddings:
    
    def __init__(self, file):
        self.embeddings = {}
        self.word_rank = {}
        for idx, line in enumerate(open(file)):
            row = line.split()
            word = row[0]
            vals = np.array([float(x) for x in row[1:]])
            self.embeddings[word] = vals
            self.word_rank[word] = idx + 1
        
    
    def get_associated_word(self, word):
        # Retrieve the GloVe vector for the given word
        word_vector = self[word]

        # Find the word most similar to the given word
        associated_word = self.find_most_similar_word(word_vector)

        return associated_word

    def find_most_similar_word(self, vector, n=1, exclude=[]):
        """
        Return the most similar word to `vector` and its similarity.

        Parameters
        ----------
        vector : str or np.array
            Input to calculate similarity against.

        n : int
            Number of results to return. Defaults to 1.

        exclude : list of str
            Do not include any words in this list in what you return.

        Returns
        -------
        tuple ('word', similarity_score)
            The top result.
        """
        if type(vector) == str:
            vector = self.embeddings[vector]

        similarities = []

        for word in self.embeddings:
            if word not in exclude:
                embedding = self.embeddings[word]
                similarity = (word, self.cosine_similarity(vector, embedding))
                similarities.append(similarity)

        sort_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        return sort_similarities[:n][0]

    def cosine_similarity(self, v1, v2):
        """
        Calculate cosine similarity between v1 and v2; these could be
        either words or numpy vectors.

        If either or both are words (e.g., type(v#) == str), replace them 
        with their corresponding numpy vectors before calculating similarity.

        Parameters
        ----------
        v1, v2 : str or np.array
            The words or vectors for which to calculate similarity.

        Returns
        -------
        float
            The cosine similarity between v1 and v2.
        """
        # >>> YOUR ANSWER HERE
        if type(v1) == str:
            v1 = self.embeddings[v1]

        if type(v2) == str:
            v2 = self.embeddings[v2]

        return (np.dot(v1, v2) / (np.sqrt(np.sum(np.square(v1))) * np.sqrt(np.sum(np.square(v2)))))
        # >>> END YOUR ANSWER
    
def telepathy_game(player1, player2, player1_word, player2_word, max_iterations=10):
        converged = False
        iteration = 0

        while not converged and iteration < max_iterations:
            # Player 1 thinks of an associated word
            player1_association = player1.get_associated_word(player1_word)

            # Player 2 thinks of an associated word
            player2_association = player2.get_associated_word(player2_word)

            # Check if the players have converged on the same word
            similarity = self.cosine_similarity(player1_association, player2_association)

            if similarity > 0.8:  # You can adjust the similarity threshold
                converged = True

                print(f"Players converged on the word: {player1_association}")

            else:
                # Update the words for the next round
                player1_word = player1_association
                
                player2_word = player2_association

            iteration += 1


if __name__ == '__main__':
    player1_file = 'wikipedia_glove/glove.6B.50d.txt'
    player2_file = 'twitter_glove/glove.twitter.27B.50d.txt'
    player1 = Embeddings(file = player1_file)
    player2 = Embeddings(file = player2_file)
    
    # Initialize starting words for both players
    player1_word = "apple"
    player2_word = "banana"

    telepathy_game(player1, player2, player1_word, player2_word)