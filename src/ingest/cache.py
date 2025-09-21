from abc import ABC, abstractmethod
import mimetypes
import os
from pathlib import Path
from typing import TextIO, BinaryIO
import shutil

from dotenv import load_dotenv


load_dotenv()


class File:
    def __init__(self, object: TextIO | BinaryIO, name):
        self.object = object # this is actually a python file object, I need a better name here
        self.name: str = name
        self._filepath: Path = None
        self._mimetype: str = None
        self.encoding: str = None
        self._rounded_filesize: int = None

    @property
    def filepath(self):
        return self._filepath
    
    @filepath.setter
    def filepath(self, filepath):
        self._filepath = filepath

    @property
    def mimetype(self):
        return self._mimetype
    
    @mimetype.setter
    def mimetype(self, mimetype):
        self._mimetype = mimetype
    
    @property
    def filesize(self):
        return self._rounded_filesize
    
    @filesize.setter
    def filesize(self, filesize):
        self._rounded_filesize = filesize


class Command(ABC):
    @abstractmethod
    def execute(self) -> bool:
        pass


class Save(Command):
    def __init__(self, file: File):
        self.file = file

    def execute(self) -> bool:
        data_directory_absolute_path = Path(os.environ.get('DATA_DIRECTORY_ABSOLUTE_PATH'))
        try:
            with open(data_directory_absolute_path / self.file.name, 'wb') as buffer:
                self.file.object.seek(0)
                shutil.copyfileobj(fsrc=self.file.object, fdst=buffer)
            self.file.filepath = data_directory_absolute_path / self.file.name
            self.file.mimetype, self.file.encoding = mimetypes.guess_type(self.file.filepath)
            self.file.filesize = self.file.filepath.stat().st_size
            return True
        except Exception as e:
            print(e)
            return False


class Remove(Command):
    def __init__(self, file: File):
        self.file = file

    def execute(self) -> bool:
        try:
            self.file.filepath.unlink()
            return True
        except Exception as e:
            print(e)
            return False


class Commander:
    def perform(self, command: Command):
        command.execute()


# Created By Amit Mahapatra