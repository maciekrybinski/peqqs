# peqqs
PEQQS: a Dataset for Probing Extractive Quantity-focused Question Answering from Scientific Literature

Question Answering (QA) and Information Retrieval (IR) play a crucial role in information-seeking pipelines implemented in many emerging \emph{AI research assistant} applications. Large Language Models (LLMs) have demonstrated exceptional effectiveness on QA tasks, with Retrieval Augmented Generation (RAG) techniques often boosting the results. However, in many of those emerging applications, the onus of conducting the actual literature search falls on the user, i.e. the user searches for the relevant literature and the LLM-based assistant extracts the solicited answers from each of the user-supplied documents. The interplay between the quality of the user-conducted search and the quality of the final results remains understudied.

In this work, we focus on a specific version of such pipeline, where users aim to obtain a specific quantity as an extractive answer (e.g., a value of a particular measurable parameter). To this end, we provide a dataset of 1031 agricultural sciences abstracts annotated with correct extractive answers. Additionally, this dataset builds on our previous work, which focused on quantity-centric search from a corpus of over 3.3M documents, which means the dataset also consists of 1130 query-document relevance judgments for 39 queries. The availability of both document-level annotations and corpus-level relevance judgments means that our dataset allows for an end-to-end evaluation of an information-seeking pipeline consisting of both literature search and the QA module. We present how our dataset can be used both for the evaluation of extractive quantity-focused QA from science literature and for exploring the impact of search on the downstream results, specifically focusing on hallucinations resulting from processing irrelevant documents with LLMs.

# CODE
The code for the example experiments on the dataset is included in the C

# DATA
Datafiles in this repository include:
- docs.json - a JSON formatted file with document contents (titles, abstracts) of relevance-judged documents.
- topics.xml - an XML formatted list of 39 information needs that can be used to formulate queries in IR experiments and prompts in generative QA experiments.
- quantities_ground_truth.json - a JSON formatted structure representing ground truth extractive answers from relevant documents.
- qrels.txt - a tab separated text file, where each line represents a document-topic binary relevance judgment (1 for relevant); first column is the position number, third position is the document ID, fourth position is the relevance of the document to the topic.

Persistent dataset repository can be found here: 

