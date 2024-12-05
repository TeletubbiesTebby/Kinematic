# FRA333: PATHMASTER KINEMATIC PROJECT
โปรเจคนี้เป็นโปรเจคสำหรับจัดทำการเคลื่อนที่ของหุ่นยนต์ เพื่อเคลื่อนที่ End-effector ไปยังตำแหน่งเป้าหมาย (Goal Points) ภายใน Workspace ที่มีสิ่งกีดขวาง โดยทั่วไปมักคำนึงถึงการหลบหลีกเฉพาะส่วนของ End-effector โดยไม่พิจารณาถึง link และ Joint ของหุ่นยนต์ โครงการนี้จึงมุ่งทำการควบคุมหุ่นยนต์ 3 DoF (RRR) เพื่อให้สามารถหลบสิ่งกีดขวางได้อย่างมีประสิทธิภาพ โดยจะใช้ Path Planning ใน Joint Space ที่คำนวณมุมของข้อต่อผ่าน A* Algorithm ในรูปแบบ Toroidal Grid และใช้ Traveling Salesman Problem (TSP) เพื่อหาลำดับการเคลื่อนที่ที่ดีที่สุด เมื่อมี Goal Points จำนวน 1-3 จุด

## จุดประสงค์โครงการ
1. เพื่อศึกษาการควบคุมหุ่นยนต์ 3 DoF (RRR) ในการหลบสิ่งกีดขวางใน Work Space เพื่อประยุกต์ความรู้ด้าน Path Planning: Route Optimization ในการค้นหาเส้นทางที่สั้นที่สุด โดยมุ่งหวังให้การเคลื่อนที่มีการหมุนของข้อต่อน้อยที่สุด  
2. เพื่อลดการใช้พลังงานและเวลาในการเคลื่อนที่ เมื่อมี Goal Points มากกว่า 1 จุด

## ขอบเขต
1. ระบบควบคุมหุ่นยนต์ 3 DoF (RRR Robot)
- ระบบควบคุมหุ่นยนต์จะคำนวณการหมุนของข้อต่อให้น้อยที่สุดในเชิงมุม  
- ระบบจะทำการคำนวณตำแหน่งของปลายหุ่นยนต์ (End Effector) ตามเป้าหมายที่กำหนด  
- ระบบจะต้องสามารถปรับเปลี่ยนตำแหน่ง Goal Points และสิ่งกีดขวางในพื้นที่งานได้

2. การตั้งค่า Input Parameters สำหรับระบบ
- ความยาวของ link หุ่นยนต์สามารถกำหนดและปรับเปลี่ยนได้  
- มุมเริ่มต้นของ link หุ่นยนต์สามารถกำหนดและปรับเปลี่ยนได้  
- จำนวนและตำแหน่ง Goal Points (ไม่เกิน 3 จุด) ที่หุ่นยนต์ต้องการไปถึง  
- ขนาดและตำแหน่งของสิ่งกีดขวางที่มีลักษณะเป็นวงกลมในพื้นที่งาน (Work Space)

3. ข้อกำหนดการคำนวณและการแสดงผล
- ระบบจะคำนวณการหมุนของข้อต่อโดยมีการลดมุมการหมุนให้น้อยที่สุด  
- ผลลัพธ์การคำนวณจะถูกแสดงผลในรูปแบบ Animation โดยใช้ Matplotlib Library  
- Goal Points ที่กำหนดต้องสามารถเข้าถึงได้โดยไม่ชนสิ่งกีดขวาง และมีท่าทางที่เป็นไปได้ในการเคลื่อนที่

4. สิ่งกีดขวางใน Work Space
- สิ่งกีดขวางใน Work Space จะมีลักษณะเป็นวงกลม 
- สิ่งกีดขวางจะไม่เคลื่อนที่และจะไม่เปลี่ยนแปลงตำแหน่ง
  
5. Tools ที่ใช้
- ใช้ Matplotlib Library สำหรับการทำ Simulation และแสดงผลการเคลื่อนที่ของหุ่นยนต์

## Input ของระบบ
**3D Robot Path**
- ค่าความยาว link (link_length) และตำแหน่งเริ่มต้นของหุ่นยนต์ (initial_link_angle)
  
**Work Space Path**
- ตำแหน่งและขนาดของสิ่งกีดขวางบน Workspace 3 มิติ (obstacle)
  
**Goal point Path**
- ตำแหน่งเริ่มต้น(start_points)
- จำนวน Goal points ที่ต้องการให้หุ่นยนต์เคลื่อนที่ไป(number_of_goals) ไม่เกิน 3 Point
- ตำแหน่งของแต่ละ Goal point (goal_point_1, goal_point_2, goal_point_3) ใน Cartesian space

## เป้าหมาย
- แสดงลำดับการเคลื่อนที่ของหุ่นยนต์จากตำแหน่ง Start ถึง Goal points ทั้งหมด โดยหมุนข้อต่อเชิงมุมให้น้อยที่สุด และสามารถหลบสิ่งกีดขวางใน Work Space ได้ 

---

## System Diagram
![Pathmaster Diagram](Image/1.png)


## Installation
download this file (RRR_Robot) and place this file in the folder to be used.


### RRR Robot

**Design**

![Pathmaster Diagram](Image/2.png)

**Class RRR Robot**

- **Init**
  - Input : link length (l1, l2, l3), init_link_angle (q1, q2, q3)
  - Output : None

- **Inverse Kinematic**
  - Input : goal_point (Px, Py, Pz)
  - Output : goal_joint_space (q1, q2, q3)

- **Forward Kinematic**
  - Input : joint_angle (q1, q2, q3)
  - Output : joint_pos_plot (P1, P2, P3, PE)

### RRR Robot - Forward Kinematic
![Pathmaster Diagram](Image/3.png)

### RRR Robot - Inverse Kinematic
![Pathmaster Diagram](Image/4.png)

![Pathmaster Diagram](Image/5.png)

---

## Prove RRR Robot - Forward Kinematics and Inverse Kinematics
**Prove ความถูกต้องของสมการ Forward Kinematic และ Inverse Kinematic ด้วยวิธีการดังนี้**
1. หา goal point ด้วยฟังก์ชัน Forward Kinematic
2. จากนั้นนำ Goal point ที่ได้ไปเข้าฟังก์ชัน Inverse Kinematic เพื่อให้ได้ Goal point ใน Joint space 
3. นำ Goal point ใน Joint space ไปเข้าฟังก์ชัน Forward Kinematic อีกครั้งเพื่อตรวจสอบความถูกต้อง โดยเปรียบเทียบกับ Goal point ที่หาได้ในขั้นตอนที่ 1

![Pathmaster Diagram](Image/6.png)

### RRR Robot - Function Plot
- สามารถกำหนดความยาวของแขนและตำแหน่งเชิงมุมของแต่ละ link ได้ 
![Pathmaster Diagram](Image/7.png)

### Obstacle 
**Function Create Obstacle**
- กำหนดให้สิ่งกีดขวางอยู่ในรูปแบบของทรงกลม(Sphere)
- สามารถกำหนดจำนวนสิ่งกีดขวาง ระบุขนาดกว้าง(X-axis) ยาว(Y-axis) สูง(Z-axis) และรัศมีของสิ่งกีดขวางได้  
![image](https://github.com/user-attachments/assets/cf50a7b4-74e6-4ec7-81ad-f075def83026)

**Function Detect Obstacle**

1.ตรวจจับการชนของข้อต่อ(joints)
  - คำนวณระยะทาง 𝑑
    โดยวัดระยะทางระหว่างพิกัดศูยน์กลางของสิ่งกีดขวางกับพิกัดของข้อต่อหุ่นยนต์ในระบบสามมิติ 
  - ตรวจสอบการชน
    หากระยะทาง 𝑑 มีค่าน้อยกว่ารัศมีของสิ่งกีดขวาง (d<=รัศมี) แสดงว่า **ข้อต่อชนกับสิ่งกีดขวาง**

2.ตรวจจับการของของแขน(links)
  - คำนวณระยะทาง 𝑑
    หาระยะทางระหว่างจุดของสิ่งกีดขวางกับ Projection เวกเตอร์ของแขนหุ่นยนต์  
  - ตรวจสอบการชน
    หากระยะทาง 𝑑 มีค่าน้อยกว่ารัศมีของสิ่งกีดขวาง (d<=รัศมี) แสดงว่า **แขนหุ่นยนต์ชนกับสิ่งกีดขวาง**
    
  ![image](https://github.com/user-attachments/assets/2bce88ff-59ed-4d65-b376-552b0e9c788d)


**ผลลัพธ์การตรวจจับ**

1. No Collision Detected (ตรวจพบว่าไม่ชนสิ่งกีดขวาง)
![image](https://github.com/user-attachments/assets/7c2628ee-a3d1-4a69-a092-15a57dbd7263)

2.Collision Detected (Joint)
- ตรวจจับการชนสิ่งกีดขวางของข้อต่อหุ่นยนต์ ในกรณีพบว่า ข้อต่อชนสิ่งกีดขวาง function จะระบุ Joint Position กับ obstacle position ที่เกิดการชนกัน
![image](https://github.com/user-attachments/assets/875160cd-74ed-4c97-9caa-7adf457d2459)

3.Collision Detected (link)
- ตรวจจับการชนสิ่งกีดขวางของแขนหุ่นยนต์ ในกรณีพบว่า แขนกลชนสิ่งกีดขวาง function จะระบุ Joint Position กับ obstacle position ที่เกิดการชนกัน
![image](https://github.com/user-attachments/assets/071704e5-4947-4824-85b4-e33a8555df26)

### 3D Occupancy Grid (Toroidal Grid)
ทำการสร้าง 3D Occupancy Grid ซึ่งเป็น Grid ใน 3 มิติ ขนาด NxNxN โดยที่ N คือแทนขนาดของมุมใน Joint space เทียบเท่าค่า -pi ถึง pi และค่าในแต่ละช่องของ Grid แทนสถานะการชน (1) หรือไม่ชนสิ่งกีดขวาง (0) เมื่อกำหนดค่าตำแหน่งเชิงมุมของทั้ง 3 joint (q1, q2, q3) โดยมีขั้นตอนการสร้าง 3D Occupancy Grid คือ
- กำหนด N แทนขนาดของมุมใน Joint space เทียบเท่าค่า -pi ถึง pi
- สร้าง 3D Occupancy Grid ขนาด NxNxN
- วนลูปใน Grid เพื่อทำการ Fill ค่า โดยทำการเช็คว่าเมื่อค่าตำแหน่งเชิงมุมของทั้ง 3 joint (q1, q2, q3) เป็นค่าต่างๆ ทุกส่วนของ	หุ่นยนต์ชนสิ่งกีดขวางหรือไม่ ถ้าไม่ทำการ Fill 0 ลง Grid ช่องนั้นๆ แต่ถ้าชนสิ่งกีดขวาง ทำการ Fill 1 ลง Grid ช่องนั้นๆ โปรแกรมจะ	ทำงานวนลูปไปเรื่อยๆ จนกระทั่ง Fill ค่าครบทุกช่องของ Grid
- Return 3D Occupancy Grid

## A* in Joint space
ใช้ A* Algorithm ใน Joint Space เพื่อหาเส้นทางที่สั้นที่สุดเชิงมุม โดยมี Input สำหรับการทำ A* Search คือ 3D Occupancy Grid และตำแหน่ง Goal points ใน Joint space (ที่ได้จาก Inverse Kinematic) โดยระบบจะทำ A* Search สำหรับทุกคู่ Sequence ของการเคลื่อนที่ เช่น ถ้ามี 3 Goal points จะทำ A* search ใน 6 กรณี ดังนี้
- Start point -> Goal point 1
- Start point -> Goal point 2
- Start point -> Goal point 3
- Goal point 1 -> Goal point 2
- Goal point 1 -> Goal point 3
- Goal point 2 -> Goal point 3

โดยการค้นหา Path Planning สำหรับแต่ละคู่ของ node ด้วย A* algorithm มีหลักการทำงาน ดังนี้

เริ่มจากการกำหนด Init Joint position และ Goal Joint position
- กำหนด Init Joint position เป็น Parent node
- จากนั้นทำการ Search หา Child Note ที่เป็นไปได้ทั้งหมดใน 3D Occupancy Grid โดยถ้า child node เป็น Goal points ให้หยุดการ Search แล้วทำการ Return Path Planning และระยะทางเชองเชิงมุมที่หุ่นยนต์ต้องเคลื่อนที่ แต่ถ้า child node ไม่ใช่ Goal points จะทำการ Search ต่อไป
- ตรวจเช็คว่า child node มีค่าเป็น 1 หรืออยู่ในท่าที่ชมสิ่งกีดขวางใน 3D Occupancy Grid หรือไม่ ถ้าใช่ให้ทำการลบ Child note ดังกล่าวทิ้ง
- คำนวณ cost ฟังก์ชันของ child note ที่เหลือทั้งหมด 

จากสมการ 

![alt text](image.png)

โดยที่
G(n) คือ current path cost หรือ cost ของการเคลื่อนที่จาก Starting point จนถึง child node (ระยะการหมุนเชิงมุมของ Joint ที่ผ่านมา)

H(n) คือ heuristic function หรือ estimated cost ในการเคลื่อนที่จาก child node ไปยัง goal point (ระยะการ	เคลื่อนที่เชิงมุมระหว่างแต่ละ joint angle จนถึง Goal point)

- จากนั้นทำการเลือก child node ที่ให้ค่า Cost funtion น้อยที่สุด เพื่อกำหนดเป็น current Parent NOde แล้วจึงทำวนไป	เรื่อยๆ จนเจอ Goal point

## Traveling Salesman Problem (TSP)**



## Animation 


### Goal Point 1
```python
goal_point.append(Point(2, 1.5, 3))
```

https://github.com/user-attachments/assets/f80f6d4a-86c0-418f-8e65-44145fa07b41
  
### Goal Point 2

```python
goal_point.append(Point(-1, 2, 4))
```
https://github.com/user-attachments/assets/79a0fe0c-2679-4569-8c1e-003e753744ab


### Goal Point 3

```python
 goal_point.append(Point(-1, 2, 3))
```
https://github.com/user-attachments/assets/bcd202e9-e549-4ea1-a5f5-81f19a300de2


## References

**RRR Robot**
- https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.researchgate.net/profile/Mohamed_Mourad_Lafifi/post/How_to_avoid_singular_configurations/attachment/59d6361b79197b807799389a/AS%253A386996594855942%25401469278586939/download/Spong%2B-%2BRobot%2Bmodeling%2Band%2BControl.pdf&ved=2ahUKEwjh4_TenpCKAxVlfGwGHZzCFWsQFnoECB8QAQ&usg=AOvVaw0SmJvmoTWd1k0O3PyRfDSI

**Obstacle avoidance path planning of 6-DOF robotic arm based on improved A∗ algorithm and artificial potential field method**
- https://www.cambridge.org/core/services/aop-cambridge-core/content/view/1d3708c34fe42078d5cf0b8f3492e8b0/s0263574723001546a.pdf/obstacle-avoidance-path-planning-of-6-dof-robotic-arm-based-on-improved-a-algorithm-and-artificial-potential-field-method.pdf

**Forward Kinematic and Inverse Kinematic**
- https://jitel.polban.ac.id/jitel/article/download/101/42

**Traveling Salesman Problem (TSP)**
- https://books.google.co.th/books?hl=th&lr=&id=vhsjbqomduic&oi=fnd&pg=pp11&dq=applegate,+d.l.,+et+al.+(2007).+the+traveling+salesman+problem:+a+computational+study.+princeton+university+press.&ots=ylcevuozc5&sig=tjqx2imqtuxvpvnu95zqpyj2qpg&redir_esc=y#v=onepage&q&f=false

**Motion planning method**
- https://books.google.co.th/books?hl=th&lr=&id=nQ7aBwAAQBAJ&oi=fnd&pg=PR9&dq=MOTION+PLANNING+IN+ROBOTICS&ots=7qjSoViLl-&sig=qFbON-dMPQR2RdbUeZdRUOTa2uU&redir_esc=y#v=onepage&q&f=false

**Motion Planning in Robotics**
- https://www.google.co.th/books/edition/Robot_Motion_Planning/nQ7aBwAAQBAJ?hl=en&gbpv=1&dq=inauthor:%22Jean-Claude+Latombe%22&printsec=frontcover
