from __future__ import annotations

import math
import sys
from importlib import metadata
from typing import Any

try:
    import numpy as np
    import soundcard as sc
except Exception as exc:
    raise SystemExit(
        "Dépendances manquantes. Exécutez : "
        "py -m pip install --upgrade soundcard soundfile numpy\n"
        f"Erreur d'import : {type(exc).__name__}: {exc}"
    ) from exc


def channel_count(device: Any) -> int:
    raw = getattr(device, "channels", 0)
    if isinstance(raw, int):
        return raw
    try:
        return len(raw)
    except Exception:
        return 0


def print_devices() -> list[Any]:
    microphones = [
        mic
        for mic in sc.all_microphones(include_loopback=False)
        if not bool(getattr(mic, "isloopback", False))
    ]
    default = sc.default_microphone()
    default_id = str(getattr(default, "id", ""))

    print("\nMicrophones Windows détectés :")
    for index, microphone in enumerate(microphones, start=1):
        marker = " [par défaut]" if str(getattr(microphone, "id", "")) == default_id else ""
        print(
            f"  {index}. {getattr(microphone, 'name', microphone)}"
            f"{marker} — {channel_count(microphone)} canal(aux)"
        )
        print(f"     id={getattr(microphone, 'id', '')}")
    return microphones


def open_and_record(microphone: Any, samplerate: int, explicit_channels: bool) -> tuple[float, float]:
    kwargs: dict[str, Any] = {"samplerate": samplerate, "exclusive_mode": False}
    if explicit_channels:
        count = max(1, min(2, channel_count(microphone)))
        kwargs["channels"] = list(range(count))
        kwargs["blocksize"] = 8192

    try:
        context = microphone.recorder(**kwargs)
    except TypeError as exc:
        if "exclusive_mode" not in str(exc):
            raise
        kwargs.pop("exclusive_mode", None)
        context = microphone.recorder(**kwargs)

    frames = max(2048, samplerate // 2)
    with context as recorder:
        data = recorder.record(numframes=frames)

    array = np.asarray(data, dtype="float32")
    if array.size == 0:
        return -60.0, -60.0
    peak = float(np.max(np.abs(array)))
    rms = float(np.sqrt(np.mean(np.square(array, dtype="float64"))))
    peak_db = -60.0 if peak <= 0 else max(-60.0, 20 * math.log10(peak))
    rms_db = -60.0 if rms <= 0 else max(-60.0, 20 * math.log10(rms))
    return peak_db, rms_db


def main() -> int:
    if sys.platform != "win32":
        print("Ce diagnostic cible Windows/WASAPI.")
        return 2

    try:
        version = metadata.version("SoundCard")
    except metadata.PackageNotFoundError:
        version = "inconnue"
    print(f"SoundCard : {version}")

    microphones = print_devices()
    if not microphones:
        print("Aucun microphone physique détecté.")
        return 2

    raw_choice = input("\nNuméro du microphone à tester [1] : ").strip() or "1"
    try:
        selected = microphones[int(raw_choice) - 1]
    except (ValueError, IndexError):
        print("Choix invalide.")
        return 2

    print(f"\nParlez pendant les essais de « {getattr(selected, 'name', selected)} ».")
    successes = 0
    for rate in (48_000, 44_100):
        for explicit in (False, True):
            profile = "canaux natifs" if not explicit else "canaux explicites"
            try:
                peak_db, rms_db = open_and_record(selected, rate, explicit)
            except Exception as exc:
                print(
                    f"[ECHEC] {rate} Hz / {profile} : "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                successes += 1
                print(
                    f"[OK]    {rate} Hz / {profile} : "
                    f"crête {peak_db:.1f} dBFS, RMS {rms_db:.1f} dBFS"
                )

    if successes == 0:
        print(
            "\nAucun profil n'a pu ouvrir ce microphone. Fermez Teams/Zoom/OBS, "
            "vérifiez l'accès Windows au microphone, puis essayez un autre périphérique."
        )
        return 1

    print("\nAu moins un profil fonctionne. Le recorder v2 réutilise ces mêmes replis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
