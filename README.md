# Car Coupling Detector

```bash
pip install -r requirements.txt

# 1. Připrav dataset
python prepare_dataset.py

# 2. Trénuj
python train.py

# 3. Evaluace s vizualizací (výsledky v eval_results/)
python evaluate.py --split val

# 4. Inference
chmod +x find_couplings
./find_couplings image1.jpg image2.jpg
```

Výstup inference: jeden integer x-souřadnice na řádek, `-1` pokud není detekce.
