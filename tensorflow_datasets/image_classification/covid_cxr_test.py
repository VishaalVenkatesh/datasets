"""covid_cxr dataset."""

from tensorflow_datasets.image_classification import covid_cxr
import tensorflow_datasets.testing as tfds_test

class CovidCxrTest(tfds_test.DatasetBuilderTestCase):
    DATASET_CLASS = covid_cxr.CovidCxr
    
    BUILDER_CONFIG_NAMES_TO_TEST = ['224']
    
    SKIP_CHECKSUMS = True
        
    SPLITS = {"train": 3,
              "test": 3,}   
    
    DL_EXTRACT_RESULT = ['train_224',  'test_224']
    
if __name__ == "__main__":
    tfds_test.test_main()
                                    
                                    
