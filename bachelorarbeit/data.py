import logging

import numpy as np
import ramanspy as rp

logger = logging.getLogger(__name__)
logger.handlers.clear()
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def load_data(file_path):
    logger.info("Starte Vorverarbeitung...")
    # Lädt die Raman-Karte vom temporären Dateipfad
    raman_image = rp.load.witec(file_path, laser_excitation=488.047)  # type: ignore
    pipeline = rp.preprocessing.Pipeline(
        [
            rp.preprocessing.despike.WhitakerHayes(),
            rp.preprocessing.denoise.Gaussian(),
            rp.preprocessing.baseline.ASLS(),
        ]
    )

    raman_image = pipeline.apply(raman_image)

    cropper_si = rp.preprocessing.misc.Cropper(region=(400, 700))
    karte_silizium = cropper_si.apply(raman_image)

    cropper_graphen = rp.preprocessing.misc.Cropper(region=(1200, 3500))
    karte_graphen = cropper_graphen.apply(raman_image)

    if karte_graphen.spectral_data.shape[-1] == 0:
        raise ValueError(
            f"Der Graphen-Bereich (1200-3500 cm⁻¹) enthält keine Datenpunkte."
        )

    si_referenz_intensitaet = karte_silizium.spectral_data.max()
    print(f"Interner Si-Standard (I_Si0) gefunden: {si_referenz_intensitaet:.2f} a.u.")

    if si_referenz_intensitaet > 0:
        karte_silizium.spectral_data /= si_referenz_intensitaet
        karte_graphen.spectral_data /= si_referenz_intensitaet

    h, w, _ = karte_graphen.spectral_data.shape
    flat_spectra = karte_graphen.spectral_data.reshape((h * w, -1))
    valid_mask_1d = np.sum(np.abs(flat_spectra), axis=1) > 1e-6

    if np.sum(valid_mask_1d) == 0:
        raise ValueError("Keine gültigen Spektren nach der Vorverarbeitung gefunden.")
    print(f"Anzahl der gültigen Spektren gefunden: {np.sum(valid_mask_1d)}")

    return valid_mask_1d, karte_silizium, karte_graphen, h, w
