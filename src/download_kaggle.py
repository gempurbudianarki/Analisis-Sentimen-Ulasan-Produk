import os
import shutil
import kagglehub

def main():
    # Set the Kaggle API Token provided by the user
    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_27b8b6ee827dca65716e69a0f0cc5749'
    
    print("[INFO] Authenticating and downloading dataset from Kaggle...")
    try:
        # Download the public Tokopedia dataset using kagglehub
        path = kagglehub.dataset_download("grikoms/tokopedia-product-reviews")
        print(f"[SUCCESS] Dataset downloaded to temporary directory: {path}")
        
        # List files in the downloaded path
        files = os.listdir(path)
        print(f"[INFO] Files found in dataset: {files}")
        
        # Copy files to our local dataset folder
        dest_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
        os.makedirs(dest_dir, exist_ok=True)
        
        for file in files:
            src_file = os.path.join(path, file)
            dest_file = os.path.join(dest_dir, file)
            if os.path.isdir(src_file):
                if os.path.exists(dest_file):
                    shutil.rmtree(dest_file)
                shutil.copytree(src_file, dest_file)
            else:
                shutil.copy2(src_file, dest_file)
            print(f"[SUCCESS] Copied {file} to {dest_file}")
            
    except Exception as e:
        print(f"[ERROR] Failed to download or copy Kaggle dataset: {e}")

if __name__ == '__main__':
    main()
