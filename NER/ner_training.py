import spacy
from spacy.training.example import Example

# Load a pre-trained English language model
try:
    nlp = spacy.load("custom_ner_model")
except:
    nlp = spacy.load("en_core_web_sm")
    
# Create a blank spaCy NER model
ner = nlp.get_pipe("ner")

labels = ["TRAIN_NUMBER", "TRAIN_NAME", "FROM", "TO", "PLATFORM_NUMBER", "DEPARTURE_TIME"]

for label in labels:
    ner.add_label(label)


train_data = [
    ("For the kind attention of passengers. Train number 12674 superfast express from Coimbatore to MG Ramachandran Central Railway Station via leave from platform number 2 at the scheduled departure time of 22 hour 50 minutes",
     {"entities": [(41, 46, "TRAIN_NUMBER"), (47, 64, "TRAIN_NAME"), (68, 80, "FROM"), (84, 121, "TO"), (129, 145, "PLATFORM_NUMBER"), (172, 185, "DEPARTURE_TIME")]}),
 
    ("For the kind attention of passengers. Train number 24680 Jan Shatabdi Express from Mumbai CST to Pune Junction via Lonavala from platform number 4 at the scheduled departure time of 09 hour 25 minutes",
     {"entities": [(41, 46, "TRAIN_NUMBER"), (47, 66, "TRAIN_NAME"), (70, 83, "FROM"), (89, 102, "TO"), (106, 116, "ROUTE"), (122, 136, "PLATFORM_NUMBER"), (167, 184, "DEPARTURE_TIME")]}),

    ("For the kind attention of passengers. Train number 13579 Sampark Kranti Express from Delhi Sarai Rohilla to Jaipur Junction via Rewari from platform number 2 at the scheduled departure time of 17 hour 50 minutes",
     {"entities": [(41, 46, "TRAIN_NUMBER"), (47, 68, "TRAIN_NAME"), (72, 91, "FROM"), (95, 110, "TO"), (114, 121, "ROUTE"), (127, 141, "PLATFORM_NUMBER"), (172, 189, "DEPARTURE_TIME")]}),

    ("For the kind attention of passengers. Train number 98765 Duronto Express from Chennai Egmore to Thiruvananthapuram Central via Coimbatore from platform number 3 at the scheduled departure time of 20 hour 15 minutes",
     {"entities": [(41, 46, "TRAIN_NUMBER"), (47, 64, "TRAIN_NAME"), (68, 83, "FROM"), (87, 114, "TO"), (118, 129, "ROUTE"), (135, 149, "PLATFORM_NUMBER"), (181, 198, "DEPARTURE_TIME")]}),

    ("For the kind attention of passengers. Train number 54321 Garib Rath Express from Kolkata Howrah Junction to Patna Junction via Mughalsarai from platform number 6 at the scheduled departure time of 16 hour 40 minutes",
     {"entities": [(41, 46, "TRAIN_NUMBER"), (47, 68, "TRAIN_NAME"), (72, 90, "FROM"), (94, 111, "TO"), (115, 127, "ROUTE"), (133, 147, "PLATFORM_NUMBER"), (182, 199, "DEPARTURE_TIME")]}),

    ("For the kind attention of passengers. Train number 98765 Deccan Queen Express from Pune Junction to Mumbai CST via Lonavala from platform number 5 at the scheduled departure time of 07 hour 30 minutes",
     {"entities": [(41, 46, "TRAIN_NUMBER"), (47, 65, "TRAIN_NAME"), (69, 82, "FROM"), (86, 98, "TO"), (102, 116, "ROUTE"), (122, 136, "PLATFORM_NUMBER"), (175, 192, "DEPARTURE_TIME")]}),

    ("For the kind attention of passengers. Train number 45678 Kanyakumari Express from Chennai Central to Kanyakumari via Nagercoil from platform number 1 at the scheduled departure time of 21 hour 05 minutes",
     {"entities": [(41, 46, "TRAIN_NUMBER"), (47, 66, "TRAIN_NAME"), (70, 83, "FROM"), (87, 99, "TO"), (103, 117, "ROUTE"), (123, 137, "PLATFORM_NUMBER"), (168, 185, "DEPARTURE_TIME")]}),

    ("For the kind attention of passengers. Train number 98765 Tejas Express from Mumbai CST to Goa Madgaon Junction via Ratnagiri from platform number 2 at the scheduled departure time of 06 hour 15 minutes",
     {"entities": [(41, 46, "TRAIN_NUMBER"), (47, 61, "TRAIN_NAME"), (65, 78, "FROM"), (82, 101, "TO"), (105, 117, "ROUTE"), (123, 137, "PLATFORM_NUMBER"), (168, 185, "DEPARTURE_TIME")]}),

    ("For the kind attention of passengers. Train number 13579 Humsafar Express from Lucknow Charbagh to Delhi Anand Vihar Terminal via Moradabad from platform number 3 at the scheduled departure time of 14 hour 20 minutes",
     {"entities": [(41, 46, "TRAIN_NUMBER"), (47, 65, "TRAIN_NAME"), (69, 87, "FROM"), (91, 115, "TO"), (119, 129, "ROUTE"), (135, 149, "PLATFORM_NUMBER"), (184, 201, "DEPARTURE_TIME")]}),

    ("For the kind attention of passengers. Train number 54321 Rajya Rani Express from Bangalore City to Mysuru Junction via Mandya from platform number 2 at the scheduled departure time of 08 hour 50 minutes",
     {"entities": [(41, 46, "TRAIN_NUMBER"), (47, 66, "TRAIN_NAME"), (70, 85, "FROM"), (89, 104, "TO"), (108, 119, "ROUTE"), (125, 139, "PLATFORM_NUMBER"), (180, 197, "DEPARTURE_TIME")]}),

    ("For the kind attention of passengers. Train number 24680 Antyodaya Express from Patna Junction to Howrah Junction via Asansol from platform number 4 at the scheduled departure time of 15 hour 30 minutes",
     {"entities": [(41, 46, "TRAIN_NUMBER"), (47, 67, "TRAIN_NAME"), (71, 86, "FROM"), (90, 110, "TO"), (114, 125, "ROUTE"), (131, 145, "PLATFORM_NUMBER"), (175, 192, "DEPARTURE_TIME")]}),
]


# Disable other pipeline components for training
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Train the NER model with the training data
with nlp.disable_pipes(*unaffected_pipes):
    optimizer = nlp.create_optimizer()
    for _ in range(120):  # You can adjust the number of iterations
        losses = {}
        for text, annotations in train_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.1, losses=losses)
        print(losses)

nlp.to_disk("custom_ner_model")



