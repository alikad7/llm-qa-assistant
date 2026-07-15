from sentence_transformers import SentenceTransformer

# مسیر مدل فعلی
model_path = "models/ms-marco-MiniLM-L-6-v2"

print(f"Loading model from {model_path} to convert...")

# لود کردن مدل (چون مدل ناقص است، از ماژول های پایه استفاده می کند)
# این همان مدلی است که HuggingFaceEmbeddings سعی می‌کند با حدس زدن بسازد
model = SentenceTransformer(model_path)

# حالا ذخیره دوباره در همان مسیر با فرمت کامل sentence-transformers
print("Saving model in standard sentence-transformers format...")
model.save(model_path)

print("✅ Conversion complete! All necessary files (modules.json, 1_Pooling, etc.) are created.")
