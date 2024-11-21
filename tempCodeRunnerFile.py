# วาดสิ่งกีดขวาง (ทรงกลม 3 มิติ)
        for obstacle in obstacles:
            u = np.linspace(0, 2 * pi, 100)  # มุมสำหรับสร้างทรงกลม
            v = np.linspace(0, pi, 100)
            x = obstacle[3] * np.outer(np.cos(u), np.sin(v)) + obstacle[0]
            y = obstacle[3] * np.outer(np.sin(u), np.sin(v)) + obstacle[1]
            z = obstacle[3] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle[2]
            ax.plot_surface(x, y, z, color='r', edgecolor='none')  # สีฟ้าพร้อมค่าความโปร่งใส