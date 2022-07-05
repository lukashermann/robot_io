import logging
import multiprocessing as mp

try:
    import pyttsx3
except ModuleNotFoundError:
    pyttsx3 = None


log = logging.getLogger(__name__)


if pyttsx3:
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
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            while True:
                text = self.queue.get()
                engine.say(text)
                engine.runAndWait()

elif pyttsx3 is None:
    class TextToSpeech:
        """
        Print text if TextToSpeech unavailable.
        """
        def say(self, x):
            print("TextToSpeech:", x)
else:
    raise ValueError


if __name__ == "__main__":
    tts = TextToSpeech()
    tts.say("Hello World!")
