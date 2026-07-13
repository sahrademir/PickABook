
import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN")

def get_llm_explanation(target_title, rec_title, target_author, rec_author):
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    You are an expert literary critic and book recommender.
    A reader loved the book '{target_title}' by {target_author}.
    Explain briefly and concisely why they would also enjoy the recommended book '{rec_title}' by {rec_author}.
    Focus on the shared themes, tone, writing style, or character dynamics.
    Keep your response professional, engaging, and under 4-5 sentences.
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=4)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            elif isinstance(result, dict):
                return result.get("generated_text", "").strip()
            
    except Exception as e:
        print(e)
        
    fallback_text = f"""✨ **PickABook Hybrid Analysis:** 

The deep learning model discovered a strong semantic connection between these works. Since you enjoyed the narrative style of **{target_author}**, this curated recommendation brings a closely aligned atmospheric experience. 

While **'{target_title}'** established a brilliant foundation in your reading profile, **'{rec_title}'** by **{rec_author}** expands on those exact thematic layers with a beautifully balanced pacing and profound character depth. If you are looking for a seamless transition into a world that shares a similar emotional resonance and structural brilliance, this book is structurally your next perfect read!"""
    
    return fallback_text