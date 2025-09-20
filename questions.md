# Questions

## Question 1
How can I load a Python file object (that contains a Markdown file) using LangChain’s `UnstructuredMarkdownLoader`?

From what I understand, the `UnstructuredMarkdownLoader` is typically designed to load Markdown files from a specified file path. It does not directly accept a Python file object as input.

Is there a way to adapt or extend the loader so that it can work with an in-memory file object (e.g., from `io.BytesIO` or `io.StringIO`) instead of requiring a file path on disk? If not, what would be the best workaround?

## Question 2
How to type hint a python file object

## Question 3
What is the difference among `TextIO`, `BinaryIO` and `IO`
`from typing import IO, TextIO, BinaryIO`

## Question 4
Difference between TypeDict and Pydantic

## Question 5
Difference between `from typing import ProtoType` and `from abc import ABC, abstractmethod`

## Question 6
What is **Chain of Responsibility** Design Pattern