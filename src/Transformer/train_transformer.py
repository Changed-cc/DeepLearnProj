from transformers import pipeline

generator = pipeline("text-generation", model="Model/transformer_finetuned")

results = generator("春", max_length=50, do_sample=True, top_k=50, temperature=0.7)

print(results[0]["generated_text"])
