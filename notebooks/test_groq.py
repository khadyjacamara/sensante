import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

temperatures = [0.0, 0.5, 1.0]

for temp in temperatures:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system",
             "content": """Tu es un assistant medical senegalais.
Tu recois un diagnostic et des donnees patient.
Explique le resultat en francais simple,
comme un medecin parlerait a son patient.
Sois rassurant mais recommande une consultation.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-meme."""},
            {"role": "user",
             "content": """Patient : Femme, 28 ans, region Dakar
Symptomes : temperature 39.5, toux, fatigue
Diagnostic du modele : paludisme (probabilite 72%)
Explique ce resultat au patient."""}
        ],
        max_tokens=200,
        temperature=temp
    )
    print(f"\n=== Temperature {temp} ===")
    print(response.choices[0].message.content)