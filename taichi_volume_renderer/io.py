import numpy as np
from .math import sigmoid

def parse_gaussian_splatting_data(ply_data):
    vertices = ply_data['vertex']
    
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T  # (N, 3)
    opacities = vertices['opacity']  # (N,)
    
    # Extracting Gaussian parameters
    # Note: Different implementations may use different naming conventions
    scales = np.vstack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']]).T  # Scaling (N, 3)
    rotations = np.vstack([vertices['rot_0'], vertices['rot_1'], vertices['rot_2'], vertices['rot_3']]).T  # Rotation quaternions (N, 4)
    
    # Extracting Spherical Harmonics (SH) coefficients:
    sh_coeffs = []
    i = 0
    while True:
        if i < 3:  # First 3 coefficients are DC terms (base color)
            key = f'f_dc_{i}'
        else:  # Remaining coefficients represent higher-frequency components
            key = f'f_rest_{i-3}'
        if not key in vertices:
            break
        sh_coeffs.append(vertices[key])
        i += 1
    
    sh_coeffs =np.array(sh_coeffs).T
    sh_coeffs = np.reshape(sh_coeffs, (sh_coeffs.shape[0], -1, 3))  # RGB triplets (3 channels) (N, -, 3)

    scales = np.exp(scales)
    sh_coeffs = sigmoid(sh_coeffs)
    opacities = sigmoid(opacities)
    
    return {
        'positions': positions,
        'opacities': opacities,
        'scales': scales,
        'rotations': rotations,
        'sh_coeffs': sh_coeffs
    }
