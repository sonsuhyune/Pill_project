### YOLO 학습시, "AttributeError: 'NoneType' object has no attribute 'shape'" 에러



- xml내의 "filename"의 text, 즉 이미지 파일 이름이 실제 이미지 파일 이름과 다른 경우 일어난다.

- 나의 경우에는,

  "257_F_7.JPG"와  "257_F_7.xml"이 있을때

  둘 다 파일 이름은 동일하지만, xml 파일 내의 filename 부분이 (라벨링 이전, 이미지 이름 변경 과정에서의 실수로 인해) "257_F_.xml" 이런식으로 되어있어 오류가 났었다.

- [darkflow issue](https://github.com/thtrieu/darkflow/issues/265)를 참고하여 data.py에 jpg를 뽑는 코드를 추가 했으나..

  위와 같은 실수가 포함된 xml파일이 많아서 그냥 코드로 xml속 filename확인하는 코드 작성

  ```python
  from xml.etree.ElementTree import parse
  import os
  import sys
  import re
  from xml.etree.ElementTree import Element, dump
  import xml.etree.ElementTree as ET
  
  
  
  li=os.listdir("train_data/pill_result/label_changed_xml/") 
  li.sort()
  for i in li:
      tree=parse('train_data/pill_result/label_changed_xml/'+i)
      root=tree.getroot()
      xml_name=root.findtext("filename")
      
      
      print(xml_name)
  ```

  