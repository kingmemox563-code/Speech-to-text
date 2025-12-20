import google.generativeai as genai
import os

class GeminiClient:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_content(self, prompt, system_instruction=None):
        try:
            if system_instruction:
                # Gemini handles system instructions differently depending on the version, 
                # but for simplicity we'll prepend it to the prompt or use the model setup if supported.
                # Here we'll combine for basic usage.
                full_prompt = f"SYSTEM: {system_instruction}\n\nUSER: {prompt}"
            else:
                full_prompt = prompt
                
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"
