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

## Learning 7

**Question**
What is **Command** Design Pattern
**Answer**
- You have a Bulb Object
- It has 2 states (On/Off)
- You have a Switch Object which will change the state of Bulb Object
- On is a command and Off is a Command, So you have a Command Abstract Object

```python
from typing import Literal

class Bulb:
    def __init__(self):
        self.state: Literal['ON', 'OFF'] = 'OFF'
    
    def toggle(self):
        if self.state == 'ON':
            self.state = 'OFF'
            print('Switch Off')
        else:
            self.state = 'ON'
            print('Switch On')

from abc import ABC, abstractmethod
class Command:
    @abstractmethod
    def execute(self, command):
        ...

class On(Command):
    def __init__(self, bulb):
        self.bulb = bulb

    def execute(self):
        self.bulb.toggle()

class Off(Command):
    def __init__(self, bulb):
        self.bulb = bulb

    def execute(self):
        self.bulb.toggle()

class Switch:
    def __init__(self, bulb: Bulb):
        self.bulb = bulb
    
    def press(self, command):
        command.execute()

bulb = Bulb()
on = On(bulb)
off = Off(bulb)

switch = Switch(bulb)

switch.press(on)
switch.press(off)
```