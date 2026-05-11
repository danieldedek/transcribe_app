# Autor: Daniel Dedek

import nemo.collections.asr as nemo_asr
import torch
from omegaconf import OmegaConf


from baseASR import BaseASR

# ASR engine vyuzivajici model nvidia/canary-180m-flash z NVIDIA NeMo
class Canary(BaseASR):
    def __init__(
        self,
        device="cpu",
        strategy="beam",
        beam_size=5,
        len_pen=1.0,
        language="en",
        use_fp16=False,
        return_hypotheses=False
    ):
        self.model_name = "nvidia/canary-180m-flash"
        self.model = None
        self.device = device
        self.strategy = strategy
        self.beam_size = beam_size
        self.len_pen = len_pen
        self.language = language
        self.use_fp16 = use_fp16
        self.return_hypotheses = return_hypotheses

    # Metoda na stazeni predtrenovanych vah modelu a nakonfigurovani dekodovaci strategie
    def download(self):
        self.model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(self.model_name)

        # Overeni dostupnosti CUDA
        device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)

        # FP16 je podporovano pouze na GPU
        if device == "cuda" and self.use_fp16:
            self.model = self.model.half()

        # Konfigurace dekodovani
        decoding_cfg = self.model.cfg.decoding
        OmegaConf.set_struct(decoding_cfg, False)
        decoding_cfg.strategy = self.strategy

        # Parametry pro "beam" strategii
        if self.strategy == "beam":
            decoding_cfg.beam.beam_size = self.beam_size
            decoding_cfg.beam.len_pen = self.len_pen

        OmegaConf.set_struct(decoding_cfg, True)

        self.model.change_decoding_strategy(decoding_cfg)

    # Metoda na prepis
    def transcribe(self, audio_path: str):
        if self.model is None:
            self.download()

        result = self.model.transcribe(
            [audio_path],
            source_lang=self.language,
            target_lang=self.language,
            return_hypotheses=self.return_hypotheses
        )

        # Vraceni celeho objektu Hypothesis
        if self.return_hypotheses:
            return result[0]
        # Vraceni standartniho vystupu
        return result[0].text
