"""
What counts as evidence — shared by the Conformità (v1) and discretionary (v2) judges.

Deliberately strict: every tender decision can be appealed, so a verdict must rest on
what the offer states, not on what can be inferred from it. Equivalence covers wording
(synonyms, other languages), never substituted or inferred content. Kept
domain-agnostic: the service evaluates tenders of any sector.
"""

EVIDENCE_RULES = """\
Criteri di prova (la valutazione deve reggere a un ricorso: nel dubbio, rigore):
- La stessa caratteristica può comparire con parole diverse, sinonimi o in un'altra lingua (le \
offerte sono spesso multilingue): valuta il significato, non la corrispondenza letterale. \
L'equivalenza riguarda SOLO le parole, mai il contenuto.
- Una dichiarazione esplicita del fornitore che il prodotto offerto possiede la caratteristica \
richiesta è evidenza valida, così come la conformità dichiarata a una norma il cui titolo o \
contenuto, riportato nelle evidenze, riguarda proprio quella caratteristica. Non pretendere dati di \
test o certificati se il requisito non li chiede espressamente.
- Quando il requisito richiama una norma, un regolamento, una direttiva o uno standard specifico, \
serve un riferimento esplicito a QUELLA norma. Una norma diversa, precedente o abrogata, oppure una \
generica "conformità alla normativa vigente", non la soddisfa: al più è soddisfazione parziale.
- NON dedurre la caratteristica: non valgono inferenze da avvertenze o precauzioni d'uso, da \
materiali simili, da studi o prove su prodotti diversi da quello offerto, da documentazione generica \
sulla categoria di prodotto.
- Se il requisito elenca più elementi richiesti insieme ("e", elenchi, "specificare A, B e C"), \
ciascuno deve essere documentato perché il requisito sia pienamente soddisfatto; se ne manca anche \
uno, la soddisfazione è parziale e la motivazione dice quale. Se invece il requisito elenca \
alternative ("o", "oppure", "in alternativa"), ne basta una, purché documentata in modo esplicito: \
non pretendere tutte le alternative.
- Nel dubbio NON considerare il requisito soddisfatto, e indica nella motivazione cosa manca."""
