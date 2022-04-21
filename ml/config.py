import os


class VnCore:
    MODEL_URI_SAVED = "https://drive.google.com/uc?export=download&id=1Jw8ivmfhqT8j1gdWy12M_I2XmSDnSeRd"
    MODEL_MD5 = "193042afa6f8f397ce38b631f15129ae"
    MODEL_PATH_SAVED = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     'models/VnCoreNLP.zip'))
    MODEL_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               'models'))
