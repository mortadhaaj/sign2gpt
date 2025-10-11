from tqdm import tqdm
import pandas as pd
import pickle
from collections import Counter
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.dediac import dediac_ar
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize

import fasttext.util

# def get_parts_of_speech(sentence: str):
#     doc = nlp(sentence)
#     lemmas, pos_tags = [], []
#     for sent in doc.sentences:
#         for w in sent.words:
#             lemmas.append(w.lemma or w.text)
#             pos_tags.append((w.text, w.upos))  # UPOS tag (NOUN, VERB, ...)
#     # align outputs with your original shape: (lemma_lower, token, pos)
#     return [( (l or "").lower(), t, p) for (l,(t,p)) in zip(lemmas, pos_tags)]

def get_parts_of_speech(sentence: str):
    """
    Returns: list[(lemma_lower, token, UPOS)]
    """
    text = sentence
    tokens = simple_word_tokenize(text)

    # Disambiguate in batch (one call)
    disambig = _mle.disambiguate(tokens)  # list of DisambiguatedWord

    out = []
    for tok, dw in zip(tokens, disambig):
        # safest: pick the top analysis if available
        if dw.analyses:
            best = dw.analyses[0]                 # ScoredAnalysis
            ana = best.analysis                   # dict with 'pos','lex','diac',...
            camel_pos = ana.get('pos')
            upos = _CAMEL2UPOS.get(camel_pos, 'X')
            lemma = (ana.get('lex') or tok).lower()
        else:
            upos = 'X'
            lemma = tok.lower()
        out.append((lemma, tok, upos))
    return out

csv_dir = 'data/isharah_selfie/dataset_random.csv'

# dict_pos = {
#     "ADJ": "adjective",
#     "ADP": "adposition",
#     "ADV": "adverb",
#     "AUX": "auxiliary verb",
#     "CONJ": "conjunction",
#     "CCONJ": "coordinating conjunction",
#     "DET": "determiner",
#     "INTJ": "interjection",
#     "NOUN": "noun",
#     "NUM": "numeral",
#     "PART": "particle",
#     "PRON": "pronoun",
#     "PROPN": "proper noun",
#     "PUNCT": "punctuation",
#     "SCONJ": "subordinating conjunction",
#     "SYM": "symbol",
#     "VERB": "verb",
#     "X": "other",
# }

# --- map CAMeL POS → UPOS (minimal, extend as needed) ---
_CAMEL2UPOS = {
    'verb': 'VERB',
    'noun': 'NOUN',
    'adj' : 'ADJ',
    'adv' : 'ADV',
    'pron': 'PRON',
    'prep': 'ADP',
    'conj': 'CONJ',     # could split into CCONJ/SCONJ if you need
    'part': 'PART',
    'det' : 'DET',
    'num' : 'NUM',
    'punc': 'PUNCT',
    'interj':'INTJ',
    'abbrev':'X',
    'res': 'X',
    None: 'X'
}

selected_vocab = ["NOUN", "NUM", "ADV", "PRON", "PROPN", "ADJ", "VERB"]

# nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma', use_gpu=True)
_mle = MLEDisambiguator.pretrained('calima-msa-r13')

df = pd.read_csv(csv_dir)
translations = df['sentence'].tolist()

dict_sentence = {}
all_lens = []
for sentence in tqdm(translations):
    pos = get_parts_of_speech(sentence)
    lems = []
    for lem, word, part in pos:
        if part in selected_vocab:
            lems.append(lem)
    all_lens.extend(lems)
    dict_sentence[sentence] = lems

fasttext.util.download_model('ar', if_exists='ignore') 
ft = fasttext.load_model('cc.ar.300.bin')

dict_lem_to_id = {l:i for i, l in enumerate(list(set(all_lens)))}


dict_processed_words = {
    "dict_lem_counter": dict(Counter(all_lens)),
    "dict_sentence": dict(dict_sentence),
    "dict_lem_to_id": dict_lem_to_id
}

# with open("data/isharah_selfie/processed_words.phx_pkl", "wb") as f:
with open("data/isharah_selfie/processed_words.pkl", "wb") as f:
    pickle.dump(dict_processed_words, f)
