# README

The most advanced version of RAG CLI implemented in this repo, will have a live version here.
Below is the list of repo, planned to be created as of now.

How Repos are named: *rag-<memory=yes/no>-<sequence number>-<data type>-<file/s>*

| Repo# | Repo Name                           | Repo Feature                                                                                                                   |
| ----- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 1     | rag-qna-01-unstructured-single      | Single File Type - Unstructured - No Memory - RAG - Q&A                                                                        |
| 2     | rag-qna-02-unstructured-multi       | Multiple File Type - Unstructured - No Memory - RAG - Q&A                                                                      |
| 3     | rag-qna-03-structured-single        | Single File Type - Structured - No Memory - RAG - Q&A                                                                          |
| 4     | rag-qna-04-structured-multi         | Multiple File Type - Structured - No Memory - RAG - Q&A                                                                        |
| 5     | rag-qna-05-mixed-multi              | Multiple File Type - Unstructured & Structured - No Memory - RAG - Q&A                                                         |
| 6     | rag-qna-06-mixed-multi-reference    | Multiple File Type - Unstructured & Structured - No Memory - RAG - \*\*With Source Reference\*\* - Q&A                         |
| 7     | rag-memory-07-mixed-multi-reference | Multiple File Type - Unstructured & Structured - \*\*With Memory\*\* - RAG - With Source Reference - With Event Watching - Q&A |



# Pluggable Multimodal RAG for Secure Knowledge Retrieval

*Note the content in below may change based on advancement of RAG application.*

## Problem Statement
- Organizations usually have **High Volume of Data** Generation daily, primarily in the form of documents, images, and videos. 
- Much of this data contains sensitive information, making **Confidential Nature of Data** a critical concern for secure access and handling.
- These documents are not static, as they undergo frequent updates, and addition and deletion, highlighting the challenge of **Continuous Updates**.
- Moreover, the data is often stored across multiple systems and platforms, leading to a **Lack of Centralization**, which makes locating specific documents a tedious process.
- Even when a document is found, users struggle with **Inefficient Search Within Documents**, as identifying the exact section containing relevant information is time-consuming.
- The situation is further complicated by **Diverse Data Formats**, ranging from structured files like .xlsx and .csv to unstructured ones like .pdf, .doc, and .txt.
- Traditional search tools fall short due to **Limited Intelligent Search Capabilities**, lacking the contextual understanding needed to interpret complex queries and deliver precise results.

## Why Pluggable?
- Within an organization, Different Teams Are Working with their own distinct datasets, often tailored to their specific functions and responsibilities. 
- These Team-Specific Data Are Not Shared Mutually, either due to confidentiality concerns or relevance boundaries. As a result, a solution built for Team 1 Should Not Have Access to Data of Team 2, even though the core problem, **Efficient Information Retrieval** remains consistent across teams.
- Building separate solutions for each team would be inefficient and redundant.
- Instead, the ideal approach is to create **One Unified System** that can be configured per team. Hence, the proposed solution is **Pluggable RAG**.

## Implementation Plan

### Phase 1 - Foundational Setup
- Begin with Document Type Identification to recognize all structured (.xlsx, .csv) and unstructured (.txt, .md, .pdf) formats used across teams. Develop Custom Data Loaders for each identified format to enable seamless and modular ingestion of documents.
- Apply LLM Based Preprocessing to clean, chunk, and semantically enrich the content for better downstream retrieval.
- Perform Vectorization and Storage by converting document chunks into embeddings and storing them in a Vector DB.
- Enable Query Handling and Retrieval to fetch the most relevant chunks from the Vector DB based on user queries. Use Response Generation via LLMs to produce accurate, context-aware answers grounded in the retrieved content.

### Phase 2 - Enhancement Phase
- **Improvement in Data Loading** by using tools like Docling, which leverage computer vision models to accurately identify and split sections within complex documents.
- **Improvement in Retrieval** by extracting metadata from documents using LLMs before storing them in the Vector DB and saving this metadata in a Graph DB for enhanced semantic search.
- **Improvement in Graph DB Updates** to ensure that metadata is dynamically refreshed whenever a document is added, modified, or deleted, maintaining consistency and relevance.
- **Improvement in Query Execution** by first searching the Graph DB for relevant metadata, then using indexing to retrieve the most useful chunks from the Vector DB, resulting in faster and more precise responses.
- **Improvement in Response Groundedness** by linking the generated answers to the original documents using metadata from the Graph DB, thereby increasing transparency and trustworthiness.

### Phase 3 - Platform Integration and Agentic RAG
- **Platform Integration** by identifying widely used organizational tools such as JIRA, GitHub, SharePoint, Outlook, and Teams Chat, and connecting to them via APIs to ingest relevant data into the RAG system.
- **Agentic Capabilities** by designing the RAG system to not only retrieve and respond but also take autonomous actions based on query context, retrieved data, and platform-specific workflows.
- **MCP Server Readiness** by preparing the system architecture to support future integration with MCP Server, enabling seamless orchestration and execution of tasks across platforms through an agentic RAG framework.

### Phase 4 - Multimodal Expansion
- **Multimodal Expansion** by extending the RAG system beyond text documents to support other data types such as images, audio, and video.
- **Multimodal Query Support** by enabling users to submit queries that can be answered using a combination of text, image, audio, or video sources, enhancing the system’s versatility and reach.

## Benefits
- **Cost-Effective Implementation**: The solution avoids expensive fine-tuning by leveraging pre-trained models making it significantly more affordable than traditional approaches.
- **Advanced Query Handling**: Capable of answering complex, multi-hop questions that require reasoning across multiple documents, showcasing its intelligent comprehension.
- **Plug-and-Play Architecture**: Easily integrates with various document formats and systems, allowing teams to adopt the solution without major infrastructure changes.
- **Enhanced Data Security**: Sensitive knowledge assets are protected by maintaining the Knowledge Graph in-house, ensuring compliance with data governance and privacy standards.
- **Improved Source Accuracy**: Metadata usage enables precise identification of source documents, increasing trust and transparency in generated responses.
- **Sustainable Long-Term Value**: Reduces operational overhead, boosts productivity, and empowers users with intelligent access to organizational knowledge, ensuring lasting impact.