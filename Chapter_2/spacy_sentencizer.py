import spacy

if __name__ == "__main__":

    nlp = spacy.load("en_core_web_sm")

    text = "Mr. Alex is a teacher. He teaches A.I. (?). Does he love his work? Of course!"
    doc = nlp(text)

    for sent in doc.sents:
        print(sent.text)