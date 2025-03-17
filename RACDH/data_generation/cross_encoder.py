from sentence_transformers import CrossEncoder


cross_encoder_model = CrossEncoder('cross-encoder/stsb-roberta-base')  # or stsb-roberta-large

def get_similarity_score(text1, text2):
    return cross_encoder_model.predict([(text1, text2)])