import os

def generate_tree(root_dir, prefix=""):
    tree_str = ""
    items = sorted(os.listdir(root_dir))
    for i, item in enumerate(items):
        path = os.path.join(root_dir, item)
        connector = "└───" if i == len(items) - 1 else "├───"
        if os.path.isdir(path):
            tree_str += f"{prefix}{connector}{item}\n"
            extension = "    " if i == len(items) - 1 else "│   "
            tree_str += generate_tree(path, prefix + extension)
        else:
            tree_str += f"{prefix}{connector.replace('─','   ')}{item}\n" if os.path.isdir(root_dir) else f"│   {item}\n"
    return tree_str

def generate_code_blocks(root_dir):
    code_str = ""
    for subdir, _, files in os.walk(root_dir):
        for file in sorted(files):
            if file.endswith(".py"):
                filepath = os.path.join(subdir, file)
                relpath = os.path.relpath(filepath, root_dir)
                code_str += f"**src/{relpath.replace(os.sep, '/')}**\n"
                code_str += "```python\n"
                with open(filepath, "r", encoding="utf-8") as f:
                    code_str += f.read()
                code_str += "\n```\n\n"
    return code_str

def main():
    src_dir = "src"
    output_file = "source_code.md"
    
    hierarchy = "## Hierarchy\n" + f"{src_dir}:.\n" + generate_tree(src_dir)
    code = "## Code\n" + generate_code_blocks(src_dir)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(hierarchy + "\n" + code)

if __name__ == "__main__":
    main()
