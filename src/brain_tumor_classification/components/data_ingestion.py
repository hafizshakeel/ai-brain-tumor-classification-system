import os
import zipfile
import gdown
import kagglehub
import shutil
from src.brain_tumor_classification import logger
from src.brain_tumor_classification.entity.config_entity import DataIngestionConfig 


class DataIngestion:
    def __init__(self, config: DataIngestionConfig, source: str):
        self.config = config
        self.source = source
        self.downloaded_path = None

    def download(self) -> str:
        """Download dataset from gdrive or kaggle."""
        try:
            # Skip if data already exists
            if os.path.exists(self.config.local_data_file):
                logger.info(f"Data already exists at {self.config.local_data_file}. Skipping download.")
                return self.config.local_data_file
            
            if self.source == "gdrive":
                self._download_from_gdrive()
            elif self.source == "kaggle":
                self._download_from_kaggle()
            else:
                raise ValueError(f"Unsupported source: {self.source}")

        except Exception as e:
            raise e
        return self.config.local_data_file

    def _download_from_gdrive(self):
        dataset_url = self.config.source_url
        zip_download_dir = self.config.local_data_file
        os.makedirs(self.config.root_dir, exist_ok=True)

        logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")
        file_id = dataset_url.split("/")[-2]
        prefix = 'https://drive.google.com/uc?/export=download&id='
        gdown.download(prefix + file_id, str(zip_download_dir), quiet=False)
        logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

    def _download_from_kaggle(self):
        dataset_slug = self.config.source_url
        os.makedirs(self.config.root_dir, exist_ok=True)

        logger.info(f"Downloading Kaggle dataset '{dataset_slug}'")
        self.downloaded_path = kagglehub.dataset_download(dataset_slug)
        logger.info(f"Dataset downloaded to cache: {self.downloaded_path}")

        # Move instead of copy (avoid duplicates)
        if os.path.exists(self.config.local_data_file):
            shutil.rmtree(self.config.local_data_file)
        shutil.move(self.downloaded_path, self.config.local_data_file)

        logger.info(f"Kaggle dataset moved to {self.config.local_data_file}")

    def extract_zip_file(self):
        """Extract if it's a zip file."""
        try:
            if str(self.config.local_data_file).endswith(".zip"):
                unzip_path = self.config.unzip_dir
                os.makedirs(unzip_path, exist_ok=True)

                with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                    zip_ref.extractall(unzip_path)

                logger.info(f"Extracted zip file to {unzip_path}")
            else:
                logger.info("No extraction needed (not a zip file).")

        except Exception as e:
            raise e

