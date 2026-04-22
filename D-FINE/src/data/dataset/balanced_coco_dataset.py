import os  
import random  
from typing import List  
from .coco_dataset import CocoDetection  
  
@register()  
class BalancedCocoDetection(CocoDetection):  
    def __init__(self, *args, max_special_per_batch=2, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.max_special_per_batch = max_special_per_batch  
        self._categorize_images()  
      
    def _categorize_images(self):  
        self.special_indices = []  
        self.normal_indices = []  
          
        for idx in range(len(self.ids)):  
            image_id = self.ids[idx]  
            file_name = self.coco.loadImgs(image_id)[0]["file_name"]  
              
            if ("coco" in file_name.lower() or   
                file_name.startswith("day_") or   
                file_name.startswith("night_")):  
                self.special_indices.append(idx)  
            else:  
                self.normal_indices.append(idx)