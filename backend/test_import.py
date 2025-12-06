try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("Import successful from langchain.text_splitter")
except ImportError:
    print("Import failed from langchain.text_splitter")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("Import successful from langchain_text_splitters")
except ImportError:
    print("Import failed from langchain_text_splitters")
