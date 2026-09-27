import os
import time
from huggingface_hub import InferenceClient

# Connect to Hugging Face
client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto"
)

# Take topic from user
topic = input("Enter a topic: ")

# Create prompt
prompt = f"""
Create 5 study flashcards about {topic}.

For each flashcard:
1. Write a question.
2. Write a short answer.
3. Keep the answers suitable for a 2nd-year
computer science student.
"""

# Start timer
start_time = time.time()

# Send request to Hugging Face model
completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# Stop timer
end_time = time.time()

# Calculate response time
latency = end_time - start_time

# Display result
print("\nGenerated Flashcards:\n")
print(completion.choices[0].message)

print(f"\nResponse time: {latency:.2f} seconds")