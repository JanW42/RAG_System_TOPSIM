"""
Projektweite Einstellungen fuer dynamische Prompt-Kontexte.

Passe PERIOD_DATE_RANGES an dein Planspiel an:
- Key: Periodennummer (int)
- Value: Dict mit:
  - start_date: Startdatum (YYYY-MM-DD oder DD.MM.YYYY)
  - end_date: Enddatum (YYYY-MM-DD oder DD.MM.YYYY)
  - end_uhrzeit: Ende der Periode am end_date (HH:MM), z. B. "10:00"
"""

# Wichtig Semesterspezifische Einstellung.
PERIOD_DATE_RANGES = {
    1: {"start_date": "13.04.2026", "end_date": "27.04.2026", "end_uhrzeit": "10:00"},
    2: {"start_date": "27.04.2026", "end_date": "11.05.2026", "end_uhrzeit": "10:00"},
    3: {"start_date": "11.05.2026", "end_date": "01.06.2026", "end_uhrzeit": "10:00"},
    4: {"start_date": "01.06.2026", "end_date": "08.06.2026", "end_uhrzeit": "10:00"},
    5: {"start_date": "08.06.2026", "end_date": "15.06.2026", "end_uhrzeit": "10:00"},
    6: {"start_date": "15.06.2026", "end_date": "22.06.2026", "end_uhrzeit": "10:00"},
}
