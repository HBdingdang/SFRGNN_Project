import os
import pathlib
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_FACE

def load_shape_from_step(step_file):
    """Load shape from a STEP file."""
    assert pathlib.Path(step_file).suffix in ['.step', '.stp', '.STEP', '.STP']
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_file))
    if status == 1:  # Check for successful reading
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape
    else:
        raise Exception("Error: Unable to read the STEP file.")

def list_faces_with_id(shape):
    """List all faces in a shape with their IDs."""
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    faces = []
    while exp.More():
        face = topods.Face(exp.Current())
        faces.append((face_id, face))
        print(f"Face ID: {face_id}, Face: {face}")
        face_id += 1
        exp.Next()
    return faces

if __name__ == "__main__":
    step_file = '/mnt/data'  # 替换为你的 STEP 文件路径
    shape = load_shape_from_step(step_file)
    faces = list_faces_with_id(shape)
