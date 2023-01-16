# Libraries
import streamlit as st
import spacy
from annotated_text import annotated_text
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re

# Design
div = '''
div.css-1fv8s86.e16nr0p34{

}

span.css-9ycgxx.exg6vvm12{
display:none;
}
small.css-7oyrr6.euu6i2w0{
display:none;
}
.css-1fttcpj.exg6vvm11:after {
  content: "Dosyaları buraya sürükleyip bırakın Limit: 200MB.";
}
button.css-5uatcg.edgvbvh10{
 visibility: hidden;
}
button.css-5uatcg.edgvbvh10:after{
    content: "Dosyaları Seçiniz";
    visibility: visible;

    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.25rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    user-select: none;
    background-color: rgb(19, 23, 32);
    border: 1px solid rgb(255, 75, 75);
}
button:not(:disabled) {
    cursor: pointer;
}
.css-8u98yl.exg6vvm0{
padding-top:2.8vh;
}
'''
st.markdown('<style>' + div + '</style>', unsafe_allow_html=True)
hide_streamlit_style = """ <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style> """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load (Cache) Model
@st.cache(show_spinner=False, allow_output_mutation=True, suppress_st_warning=True)
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("Model/")
    model = AutoModelForTokenClassification.from_pretrained("Model/")
    ner = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="first", ignore_labels= [])
    return ner

# App
def process_text(doc, selected_entities, anonymize=False):
    tokens = []
    for i in doc:
        for token in i:
            if (token['entity_group'] == "PERSON") & ("PER" in selected_entities):
                tokens.append((token['word'], "Person", "#f5d07f"))
            elif (token['entity_group'] in ["LOCATION", "LOC"]) & ("LOC" in selected_entities):
                tokens.append((token['word'], "Location", "#8ef"))
            elif (token['entity_group'] == "MONEY") & ("MONEY" in selected_entities):
                tokens.append((token['word'], "Money", "#afa"))
            # elif (token['entity_group'] == "MAHK") & ("MAHK" in selected_entities):
            #     tokens.append((token['word'], "Mahkeme", "#b693ed"))
            elif (token['entity_group'] == "TC") & ("TC" in selected_entities):
                tokens.append((token['word'], "TC", "#faa"))
            elif (token['entity_group'] == "TEL") & ("TEL" in selected_entities):
                tokens.append((token['word'], "TEL", "#90f5cd"))
            else:
                tokens.append(" " + token['word'] + " ")

    if anonymize:
        anonmized_tokens = []
        for token in tokens:
            if type(token) == tuple:
                anonmized_tokens.append((" . " * len(token[0]), token[1], token[2]))
            else:
                anonmized_tokens.append(token)
        return anonmized_tokens

    return tokens

def my_sents(doc, max_len):
    for sent in doc.sents:
        if len(sent) < max_len:
            yield sent
            continue

        # this is a long one
        offset = 0
        while offset < len(sent):
            yield sent[offset:offset+max_len]
            offset += max_len

ner = load_models()

st.sidebar.title("BERT Named Entity Recognition Model")
selected_entities = st.sidebar.multiselect(
    "Select the entities you want to detect",
    # options=["LOC", "PER", "MONEY", "MAHK", "TC", "TEL"],
    # default=["LOC", "PER", "MONEY", "MAHK", "TC", "TEL"],
    default=["LOC", "PER", "MONEY", "TC", "TEL"],
    options=["LOC", "PER", "MONEY", "TC", "TEL"],
)
# options=["LOCATION", "PERSON", "MONEY", "MAHKEME", "TC NUMARA", "TELEFON"]
options=["LOCATION", "PERSON", "MONEY", "TC NUMARA", "TELEFON"]
# colors=["#8ef","#f5d07f", "#afa", "#b693ed", "#faa","#90f5cd"]
colors=["#8ef","#f5d07f", "#afa", "#faa","#90f5cd"]

anonymize = st.sidebar.checkbox("Anonimleştirme")
colortable= ""
for i, j in zip(options, colors):
    colortable = colortable+"<tr><td style=''>"+i+"</td><td style='background-color:"+j+"; '></td></tr>"
st.sidebar.markdown("<table style='width:100%;'>"+colortable+"</table>", unsafe_allow_html=True)
selected_model = ner
text_input = st.text_area("Type a text to anonymize")

uploaded_file = st.file_uploader("or Upload a file", type=["doc", "docx", "pdf", "txt"])
if uploaded_file is not None:
    text_input = uploaded_file.getvalue()
    text_input = text_input.decode("utf-8")

if text_input:

    nlp = spacy.load('tr_floret_web_lg')
    nlp.add_pipe('sentencizer')
    text2 = re.sub(r'(\d)\s+([tT])', r'\1\2', text_input).strip()
    text2 = re.sub(r"([a-z])([A-Z])", r"\1 \2", text2)
    text2 = re.sub(r"([a-z0-9])([0-9./,])([A-Z])", r"\1\2 \3", text2)
    tokens = nlp(text2)

    nerlist = []
    for sent in tokens.sents:
        obj = my_sents(sent, 128)
        for i in obj:
            nerlist.append(ner(str(i)))
    doc = nerlist
    tokens = process_text(doc, selected_entities)
    annotated_text(*tokens)
    st.markdown("<br>",
                unsafe_allow_html=True)


    if anonymize:
        st.markdown("<h3 style='  margin: 0'>Anonimleştirilmiş metin</h3><hr style='  margin: 0'>", unsafe_allow_html=True)
        anonymized_tokens = process_text(doc, selected_entities, anonymize=anonymize)
        annotated_text(*anonymized_tokens)
