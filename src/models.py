import fasttext
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer

def load_models():

    word_embedding_model = models.Transformer(
        "deepset/gbert-large",
        max_seq_length=512
    )
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    fasttext_model = fasttext.load_model(r"C:\Users\Giuliano.DESKTOP-NPATJ24\Desktop\ptsd_classifier_new_code\notebooks\cc.de.300.bin")

    tokenizer_bert = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
    model_bert = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")

    return sentence_model, fasttext_model, tokenizer_bert, model_bert