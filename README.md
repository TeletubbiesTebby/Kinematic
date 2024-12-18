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

<div style="text-align: center;">
<img src="Image/2.png" alt="Pathmaster Diagram" width="700" />
</div>

**Class RRR Robot**

- **Init**
  - Input : link length (l1, l2, l3), init_link_angle (q1, q2, q3)
  - Output : None
```python
 RRR = RRR_Robot(l1=1.5, l2=1.5, l3=2, q1=start_joint.q1, q2=start_joint.q2, q3=start_joint.q3)
```

- **Inverse Kinematic**
  - Input : goal_point (Px, Py, Pz)
  - Output : goal_joint_space (q1, q2, q3)

- **Forward Kinematic**
  - Input : joint_angle (q1, q2, q3)
  - Output : joint_pos_plot (P1, P2, P3, PE)

### RRR Robot - Forward Kinematic

![Pathmaster Diagram](Image/3.png)

#### Code
```python
def Forward_Kinematics(self):

        P1 = Point(x=0, y=0, z=0)
        P2 = Point(x=0, y=0, z=self.l1)

        T0_1 = sp.Matrix([
            [sp.cos(self.q1), -sp.sin(self.q1), 0, 0],
            [sp.sin(self.q1), sp.cos(self.q1), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        T1_2 = sp.Matrix([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, self.l1],
            [0, 0, 0, 1]
        ])

        T2_3 = sp.Matrix([
            [sp.cos(self.q2), -sp.sin(self.q2), 0, 0],
            [sp.sin(self.q2), sp.cos(self.q2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        T3_4 = sp.Matrix([
            [sp.cos(self.q3), -sp.sin(self.q3), 0, self.l2],
            [sp.sin(self.q3), sp.cos(self.q3), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        T4_E = sp.Matrix([
            [1, 0, 0, self.l3],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        T0_4 = sp.simplify(T0_1 * T1_2 * T2_3 * T3_4)
        P3 = Point(x=T0_4[0, 3], y=T0_4[1, 3], z=T0_4[2, 3])
        # print("P3: ", P3.x, " , ", P3.y, " , ", P3.z)

        # Compute the overall transformation matrix T0_E
        T0_E = sp.simplify(T0_4 * T4_E)
        PE = Point(x=T0_E[0, 3], y=T0_E[1, 3], z=T0_E[2, 3])
        # print("PE: ", PE.x, " , ", PE.y, " , ", PE.z)

        joint_pos_plot = Joint_Pos_Plot(P1, P2, P3, PE)

        return  joint_pos_plot # (P1, P2, P3, PE)
```
#### For use
```python
    RRR = RRR_Robot(l1=1.5, l2=1.5, l3=2, q1=start_joint.q1, q2=start_joint.q2, q3=start_joint.q3)
    goal = RRR.Forward_Kinematics()
    print('Px :',goal.PE.x)
    print('Py :',goal.PE.y)
    print('Pz :',goal.PE.z)
```
#### Result
```python
Px: 1.3862096361477172
Py: 2.158893595327516
Pz: 3.7210575544202547
```

### RRR Robot - Inverse Kinematic

![Pathmaster Diagram](Image/4.png)

![Pathmaster Diagram](Image/5.png)

#### Code
```python
def Inverse_Kinematics(self, goal_point):      # goal_point  Point(x,y,z)
        Px = goal_point.x
        Py = goal_point.y
        Pz = goal_point.z

        # Solution for q1
        q1_sol = [sp.atan2(Py, Px), sp.pi + sp.atan2(Py, Px)]

        # Solutions for q2
        r = sp.sqrt(Px**2 + Py**2 + (Pz - self.l1)**2)
        cos_row = (r**2 + self.l2**2 - self.l3**2) / (2 * r * self.l2)
        sin_row = [sp.sqrt(1 - cos_row**2), -sp.sqrt(1 - cos_row**2)]

        row = [sp.atan2(sin_row[0], cos_row), sp.atan2(sin_row[1], cos_row)]
        alpha = sp.atan2(Pz - self.l1, sp.sqrt(Px**2 + Py**2))

        q2_sol = [alpha - row[0], alpha - row[1]]

        # Solutions for q3
        cos_3 = (r**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
        sin_3 = [sp.sqrt(1 - cos_3**2), -sp.sqrt(1 - cos_3**2)]

        q3_sol = [sp.atan2(sin_3[0], cos_3), sp.atan2(sin_3[1], cos_3)]

        goal_joint_space =  Joint(q1_sol[0], q2_sol[0], q3_sol[0])

        return goal_joint_space
```
#### For use
```python
  RRR = RRR_Robot(l1=1.5, l2=1.5, l3=2, q1=start_joint.q1, q2=start_joint.q2, q3=start_joint.q3)
    goal_joint_space = RRR.Inverse_Kinematics(Point(2, 1.5, 3))
    print("q1_sol :", goal_joint_space.q1)
    print("q2_sol :", goal_joint_space.q2)
    print("q3_sol :", goal_joint_space.q3)
```
#### Result
```python
q1_sol : 0.643501108793284
q2_sol : -0.148798370885615
q3_sol : 1.18639955229926
```

---

## Prove RRR Robot - Forward Kinematics and Inverse Kinematics
**Prove ความถูกต้องของสมการ Forward Kinematic และ Inverse Kinematic ด้วยวิธีการดังนี้**
1. หา goal point ด้วยฟังก์ชัน Forward Kinematic
2. จากนั้นนำ Goal point ที่ได้ไปเข้าฟังก์ชัน Inverse Kinematic เพื่อให้ได้ Goal point ใน Joint space 
3. นำ Goal point ใน Joint space ไปเข้าฟังก์ชัน Forward Kinematic อีกครั้งเพื่อตรวจสอบความถูกต้อง โดยเปรียบเทียบกับ Goal point ที่หาได้ในขั้นตอนที่ 1

![Pathmaster Diagram](Image/6.png)

## RRR Robot - Function Plot
- สามารถกำหนดความยาวของแขนและตำแหน่งเชิงมุมของแต่ละ link ได้ 
![Pathmaster Diagram](Image/7.png)

## Obstacle 
### Function Create Obstacle
- กำหนดให้สิ่งกีดขวางอยู่ในรูปแบบของทรงกลม(Sphere)
- สามารถกำหนดจำนวนสิ่งกีดขวาง ระบุขนาดกว้าง(X-axis) ยาว(Y-axis) สูง(Z-axis) และรัศมีของสิ่งกีดขวางได้  
![image](https://github.com/user-attachments/assets/cf50a7b4-74e6-4ec7-81ad-f075def83026)

### Function Detect Obstacle

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


### ผลลัพธ์การตรวจจับ

1. No Collision Detected (ตรวจพบว่าไม่ชนสิ่งกีดขวาง)
![image](https://github.com/user-attachments/assets/7c2628ee-a3d1-4a69-a092-15a57dbd7263)

2.Collision Detected (Joint)
- ตรวจจับการชนสิ่งกีดขวางของข้อต่อหุ่นยนต์ ในกรณีพบว่า ข้อต่อชนสิ่งกีดขวาง function จะระบุ Joint Position กับ obstacle position ที่เกิดการชนกัน
![image](https://github.com/user-attachments/assets/875160cd-74ed-4c97-9caa-7adf457d2459)

3.Collision Detected (link)
- ตรวจจับการชนสิ่งกีดขวางของแขนหุ่นยนต์ ในกรณีพบว่า แขนกลชนสิ่งกีดขวาง function จะระบุ Joint Position กับ obstacle position ที่เกิดการชนกัน
![image](https://github.com/user-attachments/assets/071704e5-4947-4824-85b4-e33a8555df26)

## 3D Occupancy Grid (Toroidal Grid)
ทำการสร้าง 3D Occupancy Grid ซึ่งเป็น Grid ใน 3 มิติ ขนาด MxMxM โดยกำหนดให้ M = 100 ให้ M แทนขนาดของมุมใน Joint space เทียบเท่าค่า 0 ถึง 2*pi และค่าในแต่ละช่องของ Grid แทนสถานะการชน (1) หรือไม่ชนสิ่งกีดขวาง (0) เมื่อกำหนดค่าตำแหน่งเชิงมุมของทั้ง 3 joint (q1, q2, q3) โดยมีขั้นตอนการสร้าง 3D Occupancy Grid คือ
- กำหนด N แทนขนาดของมุมใน Joint space เทียบเท่าค่า 0 ถึง 2*pi
- สร้าง 3D Occupancy Grid ขนาด MxMxM
- วนลูปใน Grid เพื่อทำการ Fill ค่า โดยทำการเช็คว่าเมื่อค่าตำแหน่งเชิงมุมของทั้ง 3 joint (q1, q2, q3) เป็นค่าต่างๆ ทุกส่วนของ	หุ่นยนต์ชนสิ่งกีดขวางหรือไม่ ถ้าไม่ทำการ Fill 0 ลง Grid ช่องนั้นๆ แต่ถ้าชนสิ่งกีดขวาง ทำการ Fill 1 ลง Grid ช่องนั้นๆ โปรแกรมจะ	ทำงานวนลูปไปเรื่อยๆ จนกระทั่ง Fill ค่าครบทุกช่องของ Grid
- Return 3D Occupancy Grid

#### code สำหรับสร้าง และบันทึก 3D Occupancy Grid (Toroidal Grid)
```python
# Create toroidal grid
def main():
    RRR = RRR_Robot(l1=1.5, l2=1.5, l3=2, q1=-1, q2=1, q3=-0.5)
    tor_grid = get_occupancy_grid(RRR, obstacles)
    
    # Save tor_grid to file
    save_tor_grid(tor_grid, 'tor_grid.npy')
```

## A* Function (in Joint space)
ใช้ A* Algorithm ใน Joint Space เพื่อหาเส้นทางที่สั้นที่สุดเชิงมุม หรือ path planning 
โดยมี Input สำหรับการทำ A* Search คือ 3D Occupancy Grid, ตำแหหน่ง start point และตำแหน่ง Goal points ใน Joint space (ที่ได้จาก Inverse Kinematic) 

โดย Function นี้จะทำการค้นหา Path Planning ด้วย A* algorithm มีหลักการทำงาน ดังนี้

เริ่มจากการกำหนด Init Joint position และ Goal Joint position
- กำหนด Init Joint position เป็น Parent node
- จากนั้นทำการ Search หา Child Note ที่เป็นไปได้ทั้งหมดใน 3D Occupancy Grid โดยถ้า child node เป็น Goal points ให้หยุดการ Search แล้วทำการ Return Path Planning และระยะทางเชองเชิงมุมที่หุ่นยนต์ต้องเคลื่อนที่ แต่ถ้า child node ไม่ใช่ Goal points จะทำการ Search ต่อไป
- ตรวจเช็คว่า child node มีค่าเป็น 1 หรืออยู่ในท่าที่ชมสิ่งกีดขวางใน 3D Occupancy Grid หรือไม่ ถ้าใช่ให้ทำการลบ Child note ดังกล่าวทิ้ง
- คำนวณ cost ฟังก์ชันของ child note ที่เหลือทั้งหมด 
  จากสมการ 

    $$
    F(n) = G(n)+H(n)
    $$

  โดยที่

  \(G(n)\) คือ current path cost หรือ cost ของการเคลื่อนที่จาก Starting point จนถึง child node (ระยะการหมุนเชิงมุมของ Joint ที่ผ่านมา)

  \(H(n)\) คือ heuristic function หรือ estimated cost ในการเคลื่อนที่จาก child node ไปยัง goal point (ระยะการ	เคลื่อนที่เชิงมุมระหว่างแต่ละ joint angle จนถึง Goal point)

- จากนั้นทำการเลือก child node ที่ให้ค่า Cost funtion น้อยที่สุด เพื่อกำหนดเป็น current Parent NOde แล้วจึงทำวนไป	เรื่อยๆ จนเจอ Goal point

#### For use
```python
tor_grid = load_tor_grid('tor_grid.npy')
start_joint = Joint(1, 1, -0.5)
goal_point = Point(2, 1.5, 3)

RRR = RRR_Robot(l1=1.5, l2=1.5, l3=2, q1=start_joint.q1, q2=start_joint.q2, q3=start_joint.q3)
goal_joint = RRR.Inverse_Kinematics(goal_point) # หา goal ใน joint space

path = astar_torus(tor_grid, start_indices, goal_indices, M)
```
#### Result
```python
path = [(10, 97, 18), (11, 96, 19), (12, 96, 20), (13, 96, 21), 
        (14, 96, 22), (15, 96, 22), (16, 96, 22), (17, 96, 22),
        (18, 96, 22), (19, 96, 22), (20, 96, 22), (21, 96, 22),
        (22, 96, 22), (23, 96, 22), (24, 96, 22), (25, 96, 22), 
        (26, 96, 22), (27, 96, 22), (28, 96, 22), (29, 96, 22), 
        (30, 96, 22), (31, 96, 22), (32, 96, 22)]
```

## Traveling Salesman Problem (TSP)

ใช้สำหรับการแก้ปัญหาการจัดเรียงลำดับ (Sequence) การเคลื่อนที่ของหุ่นยนต์ Starting point ไปยัง Goal point1, Goal point2 และ Goal point3 	เพื่อให้ได้ Sequence การเคลื่อนที่ของหุ่นยนต์ว่าควรเคลื่อนไปยัง Goal point ใดก่อนและหลัง ให้มีระยะทางเชิงมุมในการหมุนของ Joint น้อยที่สุด สำหรับประหยัดพลังงานและเวลาที่ใช้

โดยการทำ TSP Slover ในปัญหานี้ ระบบมีขั้นตอนการทำงาน ดังนี้

- Function นี้จะทำ A* Search สำหรับทุกคู่ start point และ Goal point ที่เป็นไปได้ เพื่อหา Sequence ของการเคลื่อนที่สั้นที่สุด เช่น ถ้ามี 3 Goal points จะทำ A* search ใน 6 กรณี ดังนี้
  * Start point -> Goal point 1
  * Start point -> Goal point 2
  * Start point -> Goal point 3
  * Goal point 1 -> Goal point 2
  * Goal point 1 -> Goal point 3
  * Goal point 2 -> Goal point 3

- ทำการ Brute force คำนวณระยะทางจาก Start Node ไปยังทุกเส้นทางที่ผ่านทุก Node ที่เป็นไปได้ เพื่อค้นหา Movement sequence ที่มีระยะทางเชิงมุมที่สั้นที่สุด (เลือกใช้ Brute force เพราะจากขอบเขตของโปรเจ็กต์ สามารถกำหนด Goal point ได้ไม่เกิด 3 ตำแหน่ง ทำให้มี Movement sequence ที่เป็นไปได้ทั้งหมดไม่เกิน 6 รูปแบบ ซึ่งเป็นจำนวนที่สามารถใช้ Brute force เพื่อแก้ปัญหาได้)
  - ตัวอย่างเช่น หากมี Goal point 3 จุด จะทำการค้นหาระยะทางการเคลื่อนที่เชิงมุมรวมของทั้ง 3 Joint ที่สั้นที่สุดจากทั้ง 6 sequence ดังนี้
    * start->goal1->goal2->goal3
    * start->goal1->goal3->goal2
    * start->goal2->goal1->goal3
    * start->goal2->goal3->goal1
     * start->goal3->goal1->goal2
    * start->goal3->goal2->goal1

- เมื่อค้นหาเจอ Movement sequence ที่มีระยะทางเชิงมุมที่สั้นที่สุดแล้ว จึงทำการ Return Movement sequence และ path planning ดังกล่าวออกมา เพื่อนำไปทำ Animation แสดงการเคลื่อนที่ของหุ่นยนต์ต่อไป

#### For use
```python
  tor_grid = load_tor_grid('tor_grid.npy')    # Load Grid 
  start_joint = Joint(1, 1, -0.5)

  goal_point = []
  goal_point.append(Point(2, 1.5, 3))
  goal_point.append(Point(-1, 2, 4))
  goal_point.append(Point(-1, 2, 3))

  RRR = RRR_Robot(l1=1.5, l2=1.5, l3=2, q1=start_joint.q1, q2=start_joint.q2, q3=start_joint.q3)

  posible_paths, seqence = TSP(RRR, start_joint, goal_point, tor_grid, obstacles)
```
#### Result
1. posible_paths (Example Path form Goal 2 -> Goal 3)
```python
i = 1  
j = 2  
start  = pi - atan(2) ,  -0.335889212936073 + atan(0.5*sqrt(5)) ,  0.585685543457151  
path   = [(32, 8, 9), (32, 7, 10), (32, 6, 11), (32, 5, 12), (32, 4, 13), 
          (32, 3, 14), (32, 2, 15), (32, 1, 16), (32, 0, 17),(32, 99, 18), 
          (32, 98, 19), (32, 97, 20), (32, 96, 21), (32, 96, 22)]  
total_dist = 1.5707963267948961
```
2. Optimal Moving Sequence
```python
Start   -> Goal 2
Goal 2  -> Goal 3
Goal 3  -> Goal 1
```

## Animation 
หลังจากที่ได้ Movement sequence จาก TSP Slover และ Path Planning จาก A* in Joint space ระบบจะทำการแสดง Animation การเคลื่อนที่ของหุ่นยนต์ไปยัง Goal points ต่างๆ โดยจะใช้ Forward Kinematic สำหรับคำนวณตำแหน่งของแต่ละ joint ของหุ่นยนต์ใน Cartesian space และตำแหน่งของ End-Effector เพื่อใช้สำหรับ Plot หุ่นยนต์ใน Animation ทำการ Plot หุ่นยนต์ให้เคลื่อนที่ตาม Path Planning ไปยัง Goal points ต่างๆ ตาม Movement sequence จนครบทั้งหมด แล้วจึงจบการทำงาน

### Full Project Code Demo with animation 
กำหนด Goal Point 3 Goal Point ที่ต้องการให้แขนกลเคลื่อนที่ไป 
```python
  def main():
    tor_grid = load_tor_grid('tor_grid.npy')    # Load Grid 

    start_joint = Joint(1, 1, -0.5)
    
    goal_point = []
    goal_point.append(Point(2, 1.5, 3))
    goal_point.append(Point(-1, 2, 4))
    goal_point.append(Point(-1, 2, 3))

    RRR = RRR_Robot(l1=1.5, l2=1.5, l3=2, q1=start_joint.q1, q2=start_joint.q2, q3=start_joint.q3)

    posible_paths, seqence = TSP(RRR, start_joint, goal_point, tor_grid, obstacles)

    if posible_paths != -1:
        print("**** Optimal Moving Sequence ****")
        for s in seqence:
            if(s[1] == -1):
                print("Start ->", end=" ")
            else:
                print("Goal ", s[1]+1, " ->", end=" ")
            print("Goal ", s[2]+1)

            if s[1] > s[2]:
                posible_paths[s[0]].path.reverse()
            animate_path(posible_paths[s[0]].path, obstacles, RRR , posible_paths[s[0]].start)
```
โค้ดจะคำนวณระยะทางรวมและเลือกลำดับที่ใช้ระยะทางน้อยที่สุด
```python
posible_paths = [
    # Path From Start -> Goal 1
    P2P_Path(
        -1, 0, (1, 1, -0.5),
        [
            (15, 15, 92), (14, 14, 93), (13, 13, 94), (12, 12, 95), (11, 11, 96),
            (10, 10, 97), (10, 9, 98), (10, 8, 99), (10, 7, 0), (10, 6, 1),
            (10, 5, 2), (10, 4, 3), (10, 3, 4), (10, 2, 5), (10, 1, 6),
            (10, 0, 7), (10, 99, 8), (10, 98, 9), (10, 97, 10), (10, 97, 11),
            (10, 97, 12), (10, 97, 13), (10, 97, 14), (10, 97, 15), (10, 97, 16),
            (10, 97, 17), (10, 97, 18)
        ],
        3.0787608005179976
    ),
    # Path From Start -> Goal 2
    P2P_Path(
        -1, 1, (1, 1, -0.5),
        [
            (15, 15, 92), (16, 14, 93), (17, 13, 94), (18, 12, 95), (19, 11, 96),
            (20, 10, 97), (21, 9, 98), (22, 8, 99), (23, 8, 0), (24, 8, 1),
            (25, 8, 2), (26, 8, 3), (27, 8, 4), (28, 8, 5), (29, 8, 6),
            (30, 8, 7), (31, 8, 8), (32, 8, 9)
        ],
        2.5761059759436304
    ),
    # Path From Start -> Goal 3
    P2P_Path(
        -1, 2, (1, 1, -0.5),
        [
            (15, 15, 92), (16, 14, 93), (17, 13, 94), (18, 12, 95), (19, 11, 96),
            (20, 10, 97), (21, 9, 98), (22, 8, 99), (23, 7, 0), (24, 6, 1),
            (25, 5, 2), (26, 4, 3), (27, 3, 4), (28, 2, 5), (29, 1, 6),
            (30, 0, 7), (31, 99, 8), (32, 98, 9), (32, 97, 10), (32, 96, 11),
            (32, 96, 12), (32, 96, 13), (32, 96, 14), (32, 96, 15), (32, 96, 16),
            (32, 96, 17), (32, 96, 18), (32, 96, 19), (32, 96, 20), (32, 96, 21),
            (32, 96, 22)
        ],
        4.146902302738527
    ),
    # Path From Goal 1 -> Goal 2
    P2P_Path(
        0, 1, (0.643501108793284, -0.148798370885615, 1.18639955229926),
        [
            (10, 97, 18), (11, 98, 17), (12, 99, 16), (13, 0, 15), (14, 1, 14),
            (15, 2, 13), (16, 3, 12), (17, 4, 11), (18, 5, 10), (19, 6, 9),
            (20, 7, 9), (21, 8, 9), (22, 8, 9), (23, 8, 9), (24, 8, 9),
            (25, 8, 9), (26, 8, 9), (27, 8, 9), (28, 8, 9), (29, 8, 9),
            (30, 8, 9), (31, 8, 9), (32, 8, 9)
        ],
        2.638937829015426
    ),
    # Path From Goal 1 -> Goal 3
    P2P_Path(
        0, 2, (0.643501108793284, -0.148798370885615, 1.18639955229926),
        [
            (10, 97, 18), (11, 96, 19), (12, 96, 20), (13, 96, 21), (14, 96, 22),
            (15, 96, 22), (16, 96, 22), (17, 96, 22), (18, 96, 22), (19, 96, 22),
            (20, 96, 22), (21, 96, 22), (22, 96, 22), (23, 96, 22), (24, 96, 22),
            (25, 96, 22), (26, 96, 22), (27, 96, 22), (28, 96, 22), (29, 96, 22),
            (30, 96, 22), (31, 96, 22), (32, 96, 22)
        ],
        1.6964600329384882
    ),
    # Path From Goal 2 -> Goal 3
    P2P_Path(
        1, 2, ("pi - atan(2)", "-0.335889212936073 + atan(0.5*sqrt(5))", 0.585685543457151),
        [
            (32, 8, 9), (32, 7, 10), (32, 6, 11), (32, 5, 12), (32, 4, 13),
            (32, 3, 14), (32, 2, 15), (32, 1, 16), (32, 0, 17), (32, 99, 18),
            (32, 98, 19), (32, 97, 20), (32, 96, 21), (32, 96, 22)
        ],
        1.5707963267948961
    )
]
```
จาก posible_paths ทำให้เราสามารถหาลำดับที่ดีที่สุดได้จากการวิเคราะห์เส้นทางทั้งหมดที่สามารถเคลื่อนที่ได้ ดังนี้
1. Start -> Goal 1 -> Goal 2 -> Goal 3 ใช้ระยะทาง 3.07876 + 2.63893 + 1.57079 = 7.28849
2. Start -> Goal 1 -> Goal 3 -> Goal 2 ใช้ระยะทาง 3.07876 + 1.69646 + 1.57079 = 6.34601
3. Start -> Goal 2 -> Goal 1 -> Goal 3 ใช้ระยะทาง 2.57610 + 2.63893 + 1.69646 = 6.91150
4. Start -> Goal 2 -> Goal 3 -> Goal 1 ใช้ระยะทาง 2.57610 + 1.57079 + 1.69646 = **5.84336** # Paths สั้นที่สุด เลือกใช้ Path นี้ 
5. Start -> Goal 3 -> Goal 1 -> Goal 2 ใช้ระยะทาง 4.14690 + 1.69646 + 2.63893 = 8.48230
6. Start -> Goal 3 -> Goal 2 -> Goal 1 ใช้ระยะทาง 4.14690 + 1.57079 + 2.63893 = 8.35663

ซึ่งจากการคำนวณข้างต้น เส้นทางที่ดีที่สุด คือ Start → Goal 2 → Goal 3 → Goal 1 โดยใช้ระยะทางรวม 5.84336

แขนกลจะเคลื่อนที่ตามลำดับ Optimal Moving Sequence ซึ่งได้จากการคำนวณโดยใช้ A* Algorithm ดังนี้
```python
**** Optimal Moving Sequence ****
Start   -> Goal 2
Goal 2  -> Goal 3
Goal 3  -> Goal 1
```

### Start เคลื่อนไป Goal Point 2

https://github.com/user-attachments/assets/f80f6d4a-86c0-418f-8e65-44145fa07b41
  
### Goal Point 2 เคลื่อนไป Goal Point 3

https://github.com/user-attachments/assets/79a0fe0c-2679-4569-8c1e-003e753744ab


### Goal Point 3 เคลื่อนไป Goal Point 1


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
