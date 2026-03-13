import copy
import logging


class TruncateFilter(logging.Filter):
    def __init__(self, max_length=2000):
        super().__init__()
        self.max_length = max_length

    def filter(self, record):
        # Invece di modificare il record originale,
        # creiamo una versione troncata del messaggio solo se serve
        msg = str(record.msg)
        if len(msg) > self.max_length:
            # Trucco: salviamo il messaggio originale
            original_msg = record.msg
            # Modifichiamo il record
            record.msg = msg[:self.max_length] + "... [TRUNCATED]"

            # Nota: poiché Python processa gli handler in sequenza,
            # questo potrebbe ancora influenzare altri handler se non stiamo attenti.
            # La soluzione definitiva è usare un Formatter personalizzato o
            # resettare il messaggio dopo l'emissione, ma c'è un modo più semplice...
        return True


class TruncatingFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', max_length=2000):
        super().__init__(fmt, datefmt, style)
        self.max_length = max_length

    def format(self, record):
        # Facciamo una copia leggera del record per non rovinare l'originale
        record_copy = copy.copy(record)
        msg = str(record_copy.msg)
        if len(msg) > self.max_length:
            record_copy.msg = msg[:self.max_length] + "... [TRUNCATED]"
        return super().format(record_copy)