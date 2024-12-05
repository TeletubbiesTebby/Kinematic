# PATHMASTER PROJECT

## จุดประสงค์โครงการ
1. เพื่อศึกษาการควบคุมหุ่นยนต์ 3 DoF (RRR) ในการหลบสิ่งกีดขวางใน Work Space เพื่อประยุกต์ความรู้ด้าน Path Planning: Route Optimization ในการค้นหาเส้นทางที่สั้นที่สุด โดยมุ่งหวังให้การเคลื่อนที่มีการหมุนของข้อต่อน้อยที่สุด  
2. เพื่อลดการใช้พลังงานและเวลาในการเคลื่อนที่ เมื่อมี Goal Points มากกว่า 1 จุด  

## Input ของระบบ
- Goals Points 1-3 ตำแหน่ง ที่จะต้องเคลื่อน End-effector ไป  

## เป้าหมาย
- แสดงลำดับการเคลื่อนที่ของหุ่นยนต์จากตำแหน่ง Start ถึง Goal points ทั้งหมด โดยหมุนข้อต่อให้น้อยที่สุดในเชิงมุม  

---

## System Diagram
![Pathmaster Diagram](Image/1.png)

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
**Prove ความถูกต้องของสมการ Forward Kinematic และ Inverse Kinematic ด้วยวิธีการดังนี้:**
1. หา goal point ด้วยฟังก์ชัน Forward Kinematic
2. จากนั้นนำ Goal point ที่ได้ไปเข้าฟังก์ชัน Inverse Kinematic เพื่อให้ได้ Goal point ใน Joint space 
3. นำ Goal point ใน Joint space ไปเข้าฟังก์ชัน Forward Kinematic อีกครั้งเพื่อตรวจสอบความถูกต้อง โดยเปรียบเทียบกับ Goal point ที่หาได้ในขั้นตอนที่ 1

![Pathmaster Diagram](Image/6.png)

### RRR Robot - Function Plot
![Pathmaster Diagram](Image/7.png)
