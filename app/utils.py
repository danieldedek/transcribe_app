# Autor: Daniel Dedek

from whisper import Whisper
from parakeet import Parakeet
from canary import Canary

# Globalni slovnik slouzici jako in-memory cache nactenych modelu
_model_cache = {}

# Tovarni funkce vracejici instanci ASR enginu se zadanymi parametry
def create_asr_engine(engine_name: str, device: str = "cpu", **kwargs):
    key = (engine_name, device, tuple(sorted(kwargs.items())))

    if key in _model_cache:
        return _model_cache[key]
    
    # Mapovani nazvu enginu na prislusne tridy
    engines = {
        "whisper": Whisper,
        "parakeet": Parakeet,
        "canary": Canary,
    }

    if engine_name not in engines:
        raise ValueError(f"Unknown ASR engine: {engine_name}")
    
    model = engines[engine_name](device=device, **kwargs)
    model.download()

    _model_cache[key] = model
    return model
