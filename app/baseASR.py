# Autor: Daniel Dedek

from abc import ABC, abstractmethod


# Abstraktni zakladni trida pro vsechny ASR enginy
class BaseASR(ABC):
    # Metoda na stazeni a inicializaci modelu
    @abstractmethod
    def download(self):
        pass
    
    # Metoda na prepis
    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs):
        pass
