# Symantic_Search
This is a Python code that utilizes the BERT (Bidirectional Encoder Representations from Transformers) model to perform various natural language processing tasks such as keyword extraction and article ranking based on query similarity.

## Prerequisites

Before running the code, ensure that you have the following libraries installed:

- transformers
- numpy
- torch
- scikit-learn
- nltk
- matplotlib

You can install the required libraries by running the following command:

```shell
pip install transformers numpy torch scikit-learn nltk matplotlib
```

The code also requires the BERT model and tokenizer from the Hugging Face transformers library. The code uses the 'bert-base-uncased' pre-trained model, which can be downloaded by running the following command:

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
```

## Usage

1. Install the required libraries and download the BERT model as mentioned in the prerequisites section.

2. Prepare your article file: The code expects the articles to be stored in a text file, with each article separated by a new line. The code will prompt you to upload the article file using the Google Colab file uploader.

3. Run the code: Once the article file is uploaded, you can run the code. It performs the following steps:

   - Preprocesses the articles by removing special characters, lowercasing, removing stopwords, and tokenizing the text.
   - Defines query words to search for in the articles.
   - Encodes the query words and article paragraphs using the BERT tokenizer.
   - Calculates the cosine similarity between the encoded queries and articles.
   - Ranks the articles based on similarity scores.
   - Extracts keywords from the top-ranked articles using TF-IDF.
   - Ranks the keywords by frequency and displays the top 10 keywords.

4. Visualize the top keywords: The code generates a bar chart to visualize the top keywords and their frequencies.

## Customization

You can customize the code to suit your specific needs:

- Modify the BERT model and tokenizer: If you want to use a different BERT model or tokenizer, you can change the model name and tokenizer name in the following lines of code:

  `````python
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')
  ```

- Adjust the number of top-ranked articles and keywords: By default, the code extracts keywords from the top 5 articles and displays the top 10 keywords. You can modify these values by changing the following lines of code:

  ````python
  top_articles = ranked_articles[:, -5:]  # Extract keywords from top 5 articles
  for keyword, freq in sorted_keywords[:10]:
  ```

- Customize the visualization: If you want to customize the bar chart visualization, you can modify the properties of the `plt.barh()` function in the code. For example, you can change the figure size, color, labels, title, etc.

## Acknowledgements

- This code uses the BERT model from the Hugging Face transformers library. For more information, visit the [Hugging Face documentation](https://huggingface.co/transformers/).
- The code utilizes the scikit-learn library for calculating cosine similarity and extracting keywords using TF-IDF.
- The NLTK library is used for text preprocessing, including tokenization and stop word removal.
- The matplotlib library is used for visualizing the top keywords and their frequencies.

Note: This code assumes that it is being run in a Jupyter Notebook environment such as Google Colab. If you are running the code locally, you may need to make necessary adjustments for file uploads and plotting.
```
