# Load the trained model
import spacy


custom_ner = spacy.load("custom_ner_model")

# Input transcript
transcript = "Kind attention to the passengers. Train number 12674 superfast express from Coimbatore to MG Ramachandran Central Railway Station via leave from platform number 2 at the scheduled departure time of 22 hour 50 minutes"

# Process the transcript with the custom NER model
doc = custom_ner(transcript)
print(doc)
# Extract entities and create a JSON/dict output
entities = {}
for ent in doc.ents:
    entities[ent.label_] = ent.text

print(entities)
