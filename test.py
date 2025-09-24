from pathlib import Path
import magic

path = Path(r"D:\projects\rag_genai_api\README.md")
print(path.exists())

# mime_type = magic.from_file(filename=str(path), mime=True)
# print(mime_type)

print(path.suffix)
