### xml file 읽기 & 특정 태그 값 변경 후 저장

* yolo로 학습할 데이터 라벨링 후 라벨값 변경

  

* 수정하고 싶은 xml

  ```xml
  <?xml version="1.0"?>
  
  -<annotation verified="yes">
  
  	<folder>ALL</folder>
  
  	<filename>126_F_1_R_L_1.jpg</filename>
  
  	<path>C:\Users\user\Downloads\ALL\ALL\126_F_1_R_L_1.jpg</path>
  
  
  	<source>
  
  		<database>Unknown</database>
  
  	</source>
  
  
  	-<size>
  
  		<width>2160</width>
  
  		<height>2880</height>
  
  		<depth>3</depth>
  
  	</size>
  
  	<segmented>0</segmented>
  
  
  	-<object>
  
  		<name>80</name>
  
  		<pose>Unspecified</pose>
  
  		<truncated>0</truncated>
  
  		<difficult>0</difficult>
  
  
  		-<bndbox>
  
  			<xmin>937</xmin>
  
  			<ymin>1046</ymin>
  
  			<xmax>1173</xmax>
  
  			<ymax>1183</ymax>
  
          </bndbox>
  
  	</object>
  
  </annotation>
  ```

  : 위의 파일에서 name의 값이 **숫자, 텍스트**이면 name의 값을 **"text"**로 대체

  ​                                                   **"|","-","+"**이면 name의 값을 **"line"**로 대체

  ​                                                             **한글**이면 name의 값을 **"logo"**로 대체

  

  : 작성한 코드

  ```python
  from xml.etree.ElementTree import parse
  import os
  import sys
  import re
  from xml.etree.ElementTree import Element, dump
  import xml.etree.ElementTree as ET
  
  
  
  li=os.listdir("img/for_yolo/annotaion_data") 
  #li.sort()
  for i in li:
      tree=parse('img/for_yolo/annotaion_data/'+i)
      root=tree.getroot()
      block1 = root.findall("object")
      #ee=ET.e
  
  
      for x in block1:
          #node1 = Element("suhyun")
          #print(x.findtext("name"))
          check=x.findtext("name")
          remo=x.find("name")
          #print(remo.text)
          is_english = re.compile('[-a-zA-Z]') 
          temp = is_english.findall(check) #영어 날림
          #print("_"*40)
          print(check)
          #print(i)
          #print("_"*40)
          if len(temp) == 0:
              if check.isdigit(): #숫자면 날림
                  x.remove(remo)
                  ET.SubElement(x, "name").text = "text"
                  ET.dump(x)
              
              elif check == "|" or check =="-" or check == "+": #사실상 필요없는 블럭
                  #print(remo.text)
                  #print("*"*20)
                  x.remove(remo)
                  ET.SubElement(x, "name").text = "line"
                  ET.dump(x)
              elif check != "|" and check != "-"and check != "+":
                  x.remove(remo)
                  
                  #for k in remo:
                  #    root.remove(k)
                  #    dump(root)
                  #root.remove(x)
                  ET.SubElement(x, "name").text = "logo"
                  ET.dump(x)
  
                  #node1.text="logo"+check
                  #root.append(node1)
                  #dump(node1)
              
          else:
              if check == "|" or check =="-" or check == "+":
                  #print(remo.text)
                  #print("*"*20)
                  x.remove(remo)
                  ET.SubElement(x, "name").text = "line"
                  ET.dump(x)
              else:
                  x.remove(remo)
                  ET.SubElement(x, "name").text = "text"
                  ET.dump(x)
          tree.write('img/for_yolo/changed_annotation_data/'+i)
  ```



* 위 코드 실행 후 xml 파일

  ```xml
  <?xml version="1.0"?>
  
  <annotation verified="yes">
  
  	<folder>ALL</folder>
  
  	<filename>126_F_1_R_L_1.jpg</filename>
  
  	<path>C:\Users\user\Downloads\ALL\ALL\126_F_1_R_L_1.jpg</path>
  
  
  	<source>
  
  		<database>Unknown</database>
  
  	</source>
  
  
  	<size>
  
  		<width>2160</width>
  
  		<height>2880</height>
  
  		<depth>3</depth>
  
  	</size>
  
  	<segmented>0</segmented>
  
  
  	<object>
  
  		<pose>Unspecified</pose>
  
  		<truncated>0</truncated>
  
  		<difficult>0</difficult>
  
  
  		<bndbox>
  
  			<xmin>937</xmin>
  
  			<ymin>1046</ymin>
  
  			<xmax>1173</xmax>
  
  			<ymax>1183</ymax>
  
  		</bndbox>
  
  		<name>text</name>
  
  	</object>
  
  </annotation>
  ```

  