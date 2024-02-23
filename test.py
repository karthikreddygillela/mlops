import spacy

nlp = spacy.load("en_core_web_sm")
def extract_named_entities(job_description):
    doc = nlp(job_description)
    named_entities = [ent.text for ent in doc.ents]
    return named_entities

# Example usage:
job_description = "Strong background in DevOps or DevSecOps Great experience of Cyber and Information Security Great experience of implementing Docker and / or Kubernetes Experience of Microservices security. Experience designing & delivering secure systems and tooling Excellent understanding of CI/CD, Infrastructure and Security as Code Certification such as CISSP, CEH, OSCP are ideal"
named_entities = extract_named_entities(job_description)
print(named_entities)
