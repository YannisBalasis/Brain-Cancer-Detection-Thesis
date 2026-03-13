import os, random, shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def collect_images(folder: Path):
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]

def main():
    # 👇 εδώ βάζεις τους φακέλους σου
    SRC = Path("/Users/yannisbalasis/Documents/thesis/data_multiclass")   # ο φάκελος με τις 4 κλάσεις
    DST = Path("/Users/yannisbalasis/Documents/thesis/data_multiclass_split")              # που θα φτιαχτούν train/val/test
    TRAIN, VAL, TEST = 0.7, 0.15, 0.15
    SEED = 42

    random.seed(SEED)

    classes = [d.name for d in SRC.iterdir() if d.is_dir()]
    print("Classes:", classes)

    for split in ["train", "val", "test"]:
        for cls in classes:
            (DST / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        imgs = collect_images(SRC / cls)
        random.shuffle(imgs)
        n = len(imgs)
        n_train, n_val = int(n*TRAIN), int(n*VAL)
        splits = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train+n_val],
            "test": imgs[n_train+n_val:]
        }
        print(f"{cls}: total={n} -> train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        for split, files in splits.items():
            for i, src_path in enumerate(files):
                dst_path = DST / split / cls / f"{src_path.stem}_{i}{src_path.suffix.lower()}"
                shutil.copy2(src_path, dst_path)

    print(f"OK! Split saved in {DST.resolve()}")

if __name__ == "__main__":
    main()
