import string

# Load raw descriptions
def load_doc(filename):
    with open(filename, 'r') as file:
        return file.read()

def parse_descriptions(doc):
    mapping = {}
    for line in doc.strip().split('\n'):
        tokens = line.split('\t')
        if len(tokens) != 2:
            continue
        image_id, caption = tokens
        image_id = image_id.split('#')[0]  # remove #0, #1, etc.
        caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping

def save_descriptions(descriptions, filename):
    lines = []
    for img_id, caps in descriptions.items():
        for cap in caps:
            lines.append(f"{img_id}\t{cap}")
    with open(filename, "w") as f:
        f.write("\n".join(lines))

# ==== Main ====
raw_text = load_doc("Flickr8k_text/Flickr8k.token.txt")
descriptions = parse_descriptions(raw_text)
save_descriptions(descriptions, "descriptions.txt")

print("✔ descriptions.txt created successfully.")
