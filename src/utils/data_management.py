import logging
from tqdm import tqdm
import random
import xml.etree.ElementTree as ET
import re

def processed_posts(fd_in, fd_out_train, fd_out_test, target_tag, split):
    
    line_num = 1
    
    for line in tqdm(fd_in):
        try:
            fd_out = fd_out_train if random.random() > split else fd_out_test
            attributes = ET.fromstring(line).attrib
            pid = attributes.get("Id", "")
            label = 1 if target_tag in attributes.get("Tags", "") else 0
            title = re.sub(r"\s+", " ", attributes.get("Title", "")).strip()
            body = re.sub(r"\s+", " ", attributes.get("Body", "")).strip()
            text = title + " " + body
            
            fd_out.write(f"{pid}\t{label}\t{text}\n")
            line_num += 1
                              
        except Exception as e:
            msg = f"Skipped the broken line {line_num}: {e}\n"
            logging.exception(e)          