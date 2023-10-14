Hate Speech Recognition Project Documentation
=============================================

Overview
--------

The Hate Speech Recognition project aims to classify text data as either hate speech or normal speech using a fine-tuned BERT model. The project leverages the Hugging Face "OxAISH-AL-LLM/wiki_toxic" dataset and the "google/bert_uncased_L-2_H-128_A-2" model. The implementation includes training and evaluation of the model, and finally, deploying the model using Flask and Docker.

Data
----

The dataset used in this project is sourced from the Hugging Face "OxAISH-AL-LLM/wiki_toxic" dataset. It is split into three subsets:

-   Train: 127,656 data points
-   Validation: 31,915 data points
-   Test: 63,978 data points

Model
-----

The core of this project is the use of the "google/bert_uncased_L-2_H-128_A-2" model. The model is fine-tuned to classify text into two categories: hate speech and normal speech. One additional layer is added to the model to output two values, which correspond to the classification labels.

Training
--------

The project utilizes the PyTorch Lightning framework for training the fine-tuned BERT model. Training is performed over multiple epochs. Two sets of model weights are saved for performance evaluation:

-   Model weights after one epoch
-   Model weights after five epochs

The model is trained to optimize accuracy in classifying hate speech vs. normal speech.

Performance
-----------

After training for one epoch, the model achieved an accuracy of 91.95%. This demonstrates the effectiveness of the fine-tuned BERT model in recognizing hate speech.

Deployment
----------

The fine-tuned model is deployed using the Flask framework, which serves as an API for classifying text as hate speech or normal speech. The application is containerized using Docker, allowing for easy distribution and scaling.


Getting Started
---------------

To run this project, follow these steps:

1.  Download the "OxAISH-AL-LLM/wiki_toxic" dataset from the Hugging Face dataset repository.
2.  Fine-tune the "google/bert_uncased_L-2_H-128_A-2" model using the training dataset.
3.  Implement the Flask API for the fine-tuned model.
4.  Dockerize the Flask application for easy deployment.

Dependencies
------------

Ensure that you have the following dependencies installed to run the project:

-   PyTorch
-   Transformers
-   PyTorch Lightning
-   Flask
-   Docker

Conclusion
----------

The Hate Speech Recognition project leverages state-of-the-art NLP models and frameworks to classify text data accurately. The project showcases the effectiveness of BERT-based models and demonstrates the deployment of a machine learning model using Flask and Docker, making it accessible for real-world applications.
