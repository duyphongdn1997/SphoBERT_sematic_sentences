from vncorenlp import VnCoreNLP
import os
import gdown
import nltk
from ml.config import VnCore


class SentenceSegmentation:
    """
    A tokenizer that divides a string into substrings by splitting on the specified string (defined in subclasses)
    """
    def __init__(self, input_file: str = None, text: str = None):

        self.input_file = input_file
        self.sent_segmentation = None
        self.process()
        self.text = text

    def process(self):
        if self.input_file:
            with open(self.input_file, "r") as f:
                self.text = f.read()
        self.sent_segmentation = nltk.sent_tokenize

    def get_sentence(self, text: str = None) -> list:
        """
        Return a sentence-tokenized copy of *text*,
        using NLTK's recommended sentence tokenizer

        :param text: text to split into sentences
        :return A list of sentence-tokenized
        """
        assert text or self.text, "Require a input data!"
        if text is not None:
            return self.sent_segmentation(text)
        return self.sent_segmentation(self.text)


class WordPreprocessing:
    """
    A serve, running file VNCoreNLPServe.jar on http://localhost:9000 to processing.
    Must use self.annotator.close() to kill current process, if not you can't load model again.
    """

    def __init__(self, sentence: str = None):
        self.vn_core_path = VnCore.MODEL_PATH
        self.vn_core_file = VnCore.MODEL_PATH_SAVED
        self.vn_core_uri = VnCore.MODEL_URI_SAVED
        self.vn_core_md5 = VnCore.MODEL_MD5
        self.sentence = sentence
        self.annotator = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(os.path.join(self.vn_core_path, "VnCoreNLP")):
            os.makedirs(self.vn_core_path, exist_ok=True)
            gdown.cached_download(self.vn_core_uri, self.vn_core_file,
                                  md5=self.vn_core_md5,
                                  postprocess=gdown.extractall)
        self.annotator = VnCoreNLP(os.path.join(self.vn_core_path, "VnCoreNLP/VnCoreNLP-1.1.1.jar"), port=9000,
                                   max_heap_size='-Xmx2g')

    def get_pos_tag(self, text: str = None) -> list:
        """
        :param text: A sentence needcorpus to define pos tag.
        :return: List of pos tag base on word segmentation.
        """
        assert text or self.sentence, "Require a input data!"
        if text is not None:
            result = self.annotator.pos_tag(text)
        else:
            result = self.annotator.pos_tag(self.sentence)
        return result

    def get_word_segmentation(self, text: str = None) -> list:
        """
        :param text: A sentence need to define word segmentation.
        :return: List of word segmentation of this sentence.
        """
        assert text or self.sentence, "Require a input data!"

        if text is not None:
            result = self.annotator.tokenize(text)
        else:
            result = self.annotator.tokenize(self.sentence)
        return result

    def get_ner(self, text: str = None) -> list:
        """
        :param text: A sentence need to define named entity.
        :return: List of named entities recognition base on word segmentation.
        """
        assert text or self.sentence, "Require a input data!"
        if text is not None:
            result = self.annotator.ner(text)
        else:
            result = self.annotator.ner(self.sentence)
        return result
