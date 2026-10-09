#!/usr/bin/env python3
"""Generate local-only accuracy clips and a two-hour late-hit WAV in ignored tmp/.
macOS System Voice output must never be committed or redistributed. Public text
below is authored for this test. Run after stt_fixture.py; delete tmp audio after use.
"""
import json
import subprocess
import wave
from pathlib import Path

ENGLISH = [
    "The morning train has arrived at the station. Please bring a blue notebook and a pencil for the journey.",
    "Our neighbor grows tomatoes in a small garden. Every morning she waters the plants before walking to the market.",
    "The library is closed on Sunday. We can return the borrowed books tomorrow and choose a story for the children.",
    "Please place the clean plates on the kitchen table. The soup is ready and everyone is coming inside for lunch.",
    "A gentle wind moved the leaves outside the window. The dog was sleeping beside the chair while we read quietly.",
    "We followed the narrow path beside the river. After crossing the wooden bridge we stopped to look at the mountains.",
    "The bicycle has a flat tire and a broken light. I will take it to the repair shop after breakfast.",
    "Our meeting begins after lunch in the room upstairs. Please close the door and leave your coat beside the window.",
]
GERMAN = [
    "Der Zug ist gerade am Bahnhof angekommen. Bitte bring ein blaues Heft und einen Bleistift für unsere gemeinsame Reise mit.",
    "Unsere Nachbarin pflanzt Tomaten in ihrem kleinen Garten. Jeden Morgen gießt sie die Pflanzen und geht danach zum Markt.",
    "Am Sonntag bleibt die Bibliothek geschlossen. Wir können die Bücher morgen zurückbringen und eine neue Geschichte für die Kinder aussuchen.",
    "Bitte stelle die sauberen Teller auf den Küchentisch. Die Suppe ist fertig und alle kommen jetzt zum Mittagessen herein.",
    "Ein leichter Wind bewegte die Blätter vor dem Fenster. Der Hund schlief neben dem Stuhl während wir gemeinsam ein Buch lasen.",
    "Wir folgten dem schmalen Weg am Fluss entlang. Hinter der alten Holzbrücke blieben wir stehen und schauten auf die Berge.",
    "Mein Fahrrad hat einen platten Reifen und eine kaputte Lampe. Nach dem Frühstück bringe ich es zur Werkstatt neben dem Bahnhof.",
    "Unsere Besprechung beginnt nach dem Mittagessen im oberen Zimmer. Bitte schließe die Tür und hänge deinen Mantel neben das Fenster.",
]
root = Path("tmp/stt-eval")
root.mkdir(parents=True, exist_ok=True)
rows = []
for language, voice, texts in [("en", "Karen", ENGLISH), ("en", "Samantha", ENGLISH), ("de", "Anna", GERMAN)]:
    for number, text in enumerate(texts):
        stem = root / f"{voice.lower()}-{number:02d}"
        aiff, converted, output = (stem.with_suffix(s) for s in [".aiff", ".converted.wav", ".wav"])
        subprocess.run(["say", "-v", voice, "-r", "150", "-o", str(aiff), text], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(converted)], check=True)
        with wave.open(str(converted)) as source:
            pcm = source.readframes(source.getnframes())
        assert 0 < len(pcm) <= 30 * 32000
        with wave.open(str(output), "wb") as target:
            target.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            target.writeframes(pcm)
        aiff.unlink()
        converted.unlink()
        rows.append(dict(file=str(output), language=language, voice=voice, text=text))
(root / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
# Stream silence followed by the known ten-second golden at [7190,7200).
with wave.open("tmp/stt-10.wav") as source:
    tail = source.readframes(160000)
assert len(tail) == 320000
with wave.open("tmp/stt-two-hours.wav", "wb") as target:
    target.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
    block = bytes(32000)
    for _ in range(7190):
        target.writeframesraw(block)
    target.writeframes(tail)
print("Generated local-only accuracy and late-hit fixtures; do not redistribute.")
