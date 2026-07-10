import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")

print("🧠 Connecting via HuggingFace Inference Client...")

try:
    client = InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        token=HF_TOKEN
    )

    system_prompt = "You are a book expert. Explain why 'The Secret History' and 'If We Were Villains' are similar in max 3 sentences. Focus on themes like Dark Academia and murder mystery. Answer in clear English."

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Explain the similarity."}
        ],
        max_tokens=150,
        temperature=0.7
    )

    print("\n--- Response From Llama-3 ---")
    print(response.choices[0].message.content.strip())
    print("-----------------------------")

except Exception as e:
    print(f"\n❌ Client Error: {e}")