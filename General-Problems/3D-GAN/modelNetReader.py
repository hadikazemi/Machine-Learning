import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise Exception('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = []
    for i_vert in range(n_verts):
        verts.append([float(s) for s in file.readline().strip().split(' ')])
    faces = []
    for i_face in range(n_faces):
        faces.append([int(s) for s in file.readline().strip().split(' ')][1:])
    return verts, faces

def sample_model(file_path, sample_num=50000):
    with open(file_path) as f:
        data = read_off(f)
        vertex = np.array(data[0])
        faces = np.array(data[1])

        triangles_a = vertex[faces[:, 0], :]
        triangles_b = vertex[faces[:, 1], :]
        triangles_c = vertex[faces[:, 2], :]

        a_b = triangles_b - triangles_a
        a_c = triangles_c - triangles_a

        a_bxa_c = np.cross(a_b, a_c, 1)
        tmp = np.sqrt(np.sum(a_bxa_c ** 2, axis=1)).reshape(-1, 1)
        areas = tmp / np.sum(tmp)
        normals = a_bxa_c / tmp

        sampled_tri_idx = np.random.choice(areas.shape[0], sample_num, True, np.squeeze(areas))

        sampled_a = triangles_a[sampled_tri_idx, :]
        sampled_b = triangles_b[sampled_tri_idx, :]
        sampled_c = triangles_c[sampled_tri_idx, :]

        pc_normal = normals[sampled_tri_idx, :]

        u = np.random.rand(sample_num, 1)
        v = np.random.rand(sample_num, 1) * (1 - u)
        w = 1 - u - v

        pc = u * sampled_a + v * sampled_b + w * sampled_c
        return pc, pc_normal

if __name__ == '__main__':
    pc, _ = sample_model('/media/hadi/HHD 6TB/Datasets/ModelNet/ModelNet10/ModelNet10/desk/train/desk_0021.off')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*pc.T, s=0.5)
    plt.show()
