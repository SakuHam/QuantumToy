from .EmergentDetector import EmergentDetector
from .BornDetector import BornDetector

DETECTOR_REGISTRY = {
    "emergent": EmergentDetector,
    "born": BornDetector,
}