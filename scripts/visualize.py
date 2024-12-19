# Visualization tools for embeddings and reasoning paths



import matplotlib.pyplot as plt

def visualize_embeddings(embeddings):
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:, 0], embeddings[:, 1])
    plt.show()
