import os
import shutil
import random

# Paths
input_dir = "/Users/yannisbalasis/Documents/thesis/smote_balanced_data"
output_base = "/Users/yannisbalasis/Documents/thesis/balanced_split"
splits = ['train', 'val', 'test']
split_ratio = [0.8, 0.1, 0.1]
classes = ['notumor', 'tumor']

# Δημιουργία φακέλων για κάθε split και class
for split in splits:
    for cls in classes:
        split_path = os.path.join(output_base, split, cls)
        os.makedirs(split_path, exist_ok=True)

# Διαχωρισμός και αντιγραφή
for cls in classes:
    class_path = os.path.join(input_dir, cls)
    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])

    split_images = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split in splits:
        for img in split_images[split]:
            src = os.path.join(class_path, img)
            dst = os.path.join(output_base, split, cls, img)
            shutil.copy2(src, dst)

print(" Διαχωρισμός 80/10/10 και αντιγραφή εικόνων ολοκληρώθηκε!")
print(" Νέα δομή: ", output_base)
