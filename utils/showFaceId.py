import os
import sys
import pathlib
from PyQt5.QtWidgets import QApplication, QMainWindow
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt
from OCC.Display.backend import load_backend, get_qt_modules

# 加载 Qt 后端
load_backend('qt-pyqt5')

from OCC.Display.qtDisplay import qtViewer3d


def load_shape_from_step(step_file):
    print("Loading STEP file:", step_file)
    assert pathlib.Path(step_file).suffix in ['.step', '.stp', '.STEP', '.STP']
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_file))
    if status == 1:  # Check for successful reading
        print("STEP file loaded successfully.")
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape
    else:
        raise Exception("Error: Unable to read the STEP file.")


def list_faces_with_id(shape):
    print("Listing faces with IDs.")
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    faces = []
    while exp.More():
        face = topods.Face(exp.Current())
        faces.append((face_id, face))
        face_id += 1
        exp.Next()
    return faces


def get_face_center(face):
    properties = BRep_Tool.Surface(face)
    u_min, u_max, v_min, v_max = properties.Bounds()
    u_center = (u_min + u_max) / 2
    v_center = (v_min + v_max) / 2
    point = properties.Value(u_center, v_center)
    return point


class PatchedMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1024, 768)
        self.setWindowTitle("3D Viewer")


def init_display():
    print("Initializing display.")
    try:
        QtCore, QtGui, QtWidgets, QtOpenGL = get_qt_modules()
        app = QtWidgets.QApplication.instance()
        if not app:
            app = QtWidgets.QApplication([])

        viewer = qtViewer3d()
        window = PatchedMainWindow()
        window.setCentralWidget(viewer)
        window.show()

        return viewer._display, app.exec_
    except Exception as e:
        print(f"Error during display initialization: {e}", file=sys.stderr)
        raise


def display_shape_with_face_ids(step_file):
    try:
        shape = load_shape_from_step(step_file)
        faces = list_faces_with_id(shape)

        display, start_display = init_display()

        display.DisplayShape(shape, update=True)

        for face_id, face in faces:
            center = get_face_center(face)
            pnt = center.Coord()
            display.DisplayMessage(gp_Pnt(pnt[0], pnt[1], pnt[2]), str(face_id))

        start_display()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    step_file = '/mnt/data.step'  # 替换为你的 STEP 文件路径
    display_shape_with_face_ids(step_file)
