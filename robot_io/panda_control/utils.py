import multiprocessing as mp
import logging
log = logging.getLogger(__name__)


class TextToSpeech:
    def __init__(self):
        self.queue = mp.Queue()
        self.process = mp.Process(target=self.tts_worker, name="TTS_worker")
        self.process.daemon = True
        self.process.start()

    def say(self, text):
        log.info(text)
        self.queue.put(text)

    def tts_worker(self):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        while True:
            text = self.queue.get()
            engine.say(text)
            engine.runAndWait()
