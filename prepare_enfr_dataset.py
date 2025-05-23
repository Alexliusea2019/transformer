from datasets import load_dataset
import os

def prepare_enfr_dataset(output_dir="data", num_samples=50000):
    # Load the dataset
    dataset = load_dataset("opus_books", "en-fr", split="train")

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the output files
    with open(os.path.join(output_dir, "train.en"), "w", encoding="utf-8") as f_en, \
         open(os.path.join(output_dir, "train.fr"), "w", encoding="utf-8") as f_fr:

        for example in dataset.select(range(min(num_samples, len(dataset)))):
            en = example["translation"]["en"].strip().replace("\n", " ")
            fr = example["translation"]["fr"].strip().replace("\n", " ")

            if en and fr:
                f_en.write(en + "\n")
                f_fr.write(fr + "\n")

    print(f"✅ Saved {num_samples} English–French sentence pairs to '{output_dir}/train.en' and '{output_dir}/train.fr'.")

if __name__ == "__main__":
    prepare_enfr_dataset()