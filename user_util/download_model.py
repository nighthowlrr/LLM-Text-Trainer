import os
import argparse
import shutil
from huggingface_hub import snapshot_download
from code_util.file_helpers import pick_folder


def main():
    parser = argparse.ArgumentParser(description="Download a HuggingFace model to a local directory.")
    parser.add_argument("--model_id", type=str, required=False,
        help="HuggingFace model id (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--save_dir", type=str, required=False,
        help="Where to save the model")
    parser.add_argument("--force_download", action="store_true",
        help="If set, automatically overwrite existing files without asking")
    args = parser.parse_args()


    # Prompt if not provided
    model_id = args.model_id or input("Enter Hugging Face model id (e.g. Qwen/Qwen2.5-1.5B-Instruct): ").strip()
    given_save_dir = args.save_dir or pick_folder("Select destination folder")
    save_dir = given_save_dir + f"/{model_id.split("/")[-1]}"

    # Warn if directory exists
    if os.path.exists(save_dir) and not args.force_download:
        choice = input(f"WARNING: Model directory already exists at {save_dir}. Overwrite and delete existing? (y/n): ").strip().lower()
        if choice != "y":
            print("Download cancelled")
            return
        else:
            print(f"Removing existing folder: {save_dir}")
            shutil.rmtree(save_dir) # Clean delete
            args.force_download = True
    elif os.path.exists(save_dir) and args.force_download:
        print(f"Removing existing folder (force): {save_dir}")
        shutil.rmtree(save_dir)

    print(f"\nDownloading {model_id} to {save_dir} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=save_dir,
        resume_download=False, # always fresh when overwriting
        force_download=True
    )
    print(f"Download complete! Model saved at: {save_dir}")


if __name__ == "__main__":
    main()