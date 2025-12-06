from langchain_huggingface import HuggingFaceEmbeddings

try:
    print("Attempting to load model: sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Model loaded successfully!")
    
    print("Testing embedding generation...")
    vector = embeddings.embed_query("This is a test.")
    print(f"✅ Generated vector of length: {len(vector)}")
except Exception as e:
    print(f"❌ Error: {e}")
