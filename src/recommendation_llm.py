import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")

def get_llm_explanation(target_title, rec_title, target_author, rec_author):
    """
    İki kitap arasındaki anlamsal benzerliği Llama-3 ile dinamik olarak açıklar.
    """
    try:
        client = InferenceClient(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            token=HF_TOKEN
        )

        # Kullanıcının aradığı kitap ile önerilen kitabı dinamik olarak prompt'a gömüyoruz
        system_prompt = (
            f"You are a book expert. Explain why the book '{rec_title}' by {rec_author} "
            f"was recommended to a user who enjoyed reading '{target_title}' by {target_author}. "
            f"Focus on shared genres, atmospheric themes, and writing style in maximum 3 sentences. "
            f"Answer in clear, fluid English."
        )

        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Explain the semantic connection between these two books."}
            ],
            max_tokens=150,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Explanation temporarily unavailable: {str(e)}"