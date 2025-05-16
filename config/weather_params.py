from enum import Enum

class WeatherParam(str, Enum):
    """Keys für Geosphere-Austria-API und dazugehörige Metadaten."""
    TL   = 'tl'
    RF   = 'rf'
    FF   = 'ff'
    FFX  = 'ffx'
    CGLO = 'cglo'
    SO_H = 'so_h'
    RR   = 'rr'
    RRM  = 'rrm'
    TB10 = 'tb10'
    TB20 = 'tb20'

    @property
    def label(self) -> str:
        labels = {
            'tl':   'Lufttemperatur (2 m)',
            'rf':   'Relative Feuchte (%)',
            'ff':   'Windgeschwindigkeit (m/s)',
            'ffx':  'Windspitzen (m/s)',
            'cglo': 'Globalstrahlung (W/m²)',
            'so_h': 'Sonnenscheindauer (h)',
            'rr':   'Niederschlag (mm)',
            'rrm':  'Niederschlagsdauer (min)',
            'tb10': 'Bodentemperatur –10 cm (°C)',
            'tb20': 'Bodentemperatur –20 cm (°C)',
        }
        return labels[self.value]

    @classmethod
    def keys(cls) -> list[str]:
        return [member.value for member in cls]

    @classmethod
    def items(cls) -> list[tuple[str,str]]:
        """Liste von (key, label)-Tupeln."""
        return [(member.value, member.label) for member in cls]