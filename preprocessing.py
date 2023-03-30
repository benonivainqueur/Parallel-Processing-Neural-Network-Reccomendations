import re
import string
import spacy
from string import digits
nlp = spacy.load('en_core_web_sm')
# nlp.disable_pipe('ner')
# nlp.disable_pipe('parser')


def remove_street_numbers(text):
    return re.sub(r'(?:\d+ )([A-Z])', r'\1', text)


def remove_level_number(s):
    return re.sub(r'(?:Level )\d+', "Level ", s)


def remove_note_number(s):
    return re.sub(r'(?:Note )\d+', "Note  ", s)


def lemmatize(s, parser= nlp):
    doc = parser(s)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])


# Clean text for the dataframe

def clean_text(text):
    punc_list = list(string.punctuation.replace('.', '').replace('-', ''))  # except dots and dashes
    translator = text.maketrans(dict.fromkeys(punc_list, " "))
    cleantext = text.lower().translate(translator)
    ## clear off numbers and normalize spaces between words
    ## and lowercase it
    cleantext = " ".join([s for s in cleantext.split(" ") if s.strip() != ""]).lower()
    ## remove any non-printable (non-ascii) characters in the text
    printable = set(string.printable)
    cleantext = list(filter(lambda x: x in printable, cleantext))
    cleantext = "".join(cleantext)
    ## remove roman numberals from string which
    ## are not in brackets
    toremove = [' ii ', ' iii ', ' iv ', ' v ', ' vi ', ' vii ', ' viii ', ' ix ', ' x ', '!', '@', '#', '$', '%', '^',
                '&', '*', '$.']
    text_array = cleantext.split("\s+")
    cleantext = [word.strip() for word in text_array if word not in toremove]
    cleantext = " ".join(cleantext)

    cleantext = re.sub(' +', ' ', cleantext)
    return cleantext.strip()


def preprocess_column(df_data, column_name, do_lemmatize=True, no_stopwords=True):
    tqdm.pandas()
    df_data = df_data[pd.notnull(df_data[column_name])]
    df_data['temp'] = df_data[column_name].progress_apply(
        lambda x: remove_all_tables(x))
    df_data['readable_text'] = df_data['temp'].progress_apply(lambda x: get_readable_text(x))
    df_data.drop(['temp'], axis=1, inplace=True)
    df_data['processed_value'] = df_data['readable_text'].progress_apply(lambda x: clean_text(x))
    if do_lemmatize:
        parser = spacy.load('en', disable=['parser', 'ner'])
        df_data['processed_value'] = df_data['processed_value'].progress_apply(
            lambda x: lemmatize(x, parser))
    if no_stopwords:
        df_data['processed_value'] = df_data['processed_value'].progress_apply(
            lambda x: ' '.join([word for word in x.split() if word not in (text.ENGLISH_STOP_WORDS)]))
    df_data = df_data[pd.notnull(df_data[column_name])]
    return df_data

def remove_num_sp_chr(sentence):
    sentence = re.sub(r"[^a-zA-Z0-9 ]", "", sentence)
    remove_digits = str.maketrans('', '', digits)
    list = [i.translate(remove_digits) for i in sentence.split()]
    return " ".join(list)
