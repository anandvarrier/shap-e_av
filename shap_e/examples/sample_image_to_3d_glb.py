import torch
import os
import pygltflib
import numpy as np
import io
import base64
from PIL import Image

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images
from shap_e.util.image_util import load_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
model = load_model('image300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

batch_size = 4
guidance_scale = 3.0

# To get the best result, you should remove the background and show only the object of interest to the model.
image = load_image("example_data/corgi.png")

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(images=[image] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)

render_mode = 'nerf' # you can change this to 'stf' for mesh rendering
size = 64 # this is the size of the renders; higher values take longer to render.

cameras = create_pan_cameras(size, device)

# Create the glTF scene
glb = pygltflib.GLTF2()

# Create a buffer for the images
buffer_data = b''.join([open("example_data/corgi.png", 'rb').read()])
buffer = pygltflib.Buffer(byteLength=len(buffer_data))
glb.buffers.append(buffer)

# Convert the image data to bytes and store it in the buffer
buffer.data = np.frombuffer(buffer_data, dtype=np.uint8).tobytes()

# Create an image and texture
image = pygltflib.Image(bufferView=pygltflib.BufferView(buffer=0, byteOffset=0), mimeType="image/png")
texture = pygltflib.Texture(source=image)
glb.textures.append(texture)
glb.images.append(image)

# Create a material using the texture
material = pygltflib.Material(pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(baseColorTexture=pygltflib.TextureInfo(index=0)))
glb.materials.append(material)

# Create a mesh with a single plane and apply the material
plane_vertices = np.array([[-1.0, -1.0, 0], [1.0, -1.0, 0], [1.0, 1.0, 0], [-1.0, 1.0, 0]])
plane_faces = np.array([[0, 1, 2], [0, 2, 3]])
plane = pygltflib.Mesh(primitives=[pygltflib.Primitive(mode=4, attributes={'POSITION': 0}, indices=0, material=0)])
glb.meshes.append(plane)

# Create a buffer view for the plane vertices and faces
plane_vertices_buffer_view = pygltflib.BufferView(buffer=0, byteOffset=0)
plane_faces_buffer_view = pygltflib.BufferView(buffer=0, byteOffset=len(plane_vertices) * 4)

glb.bufferViews.append(plane_vertices_buffer_view)
glb.bufferViews.append(plane_faces_buffer_view)

# Create an accessor for the plane vertices and faces
plane_vertices_accessor = pygltflib.Accessor(bufferView=0, byteOffset=0, componentType=pygltflib.ComponentType.FLOAT, count=len(plane_vertices),
                                            type=pygltflib.Type.VEC3, max=np.max(plane_vertices, axis=0).tolist(), min=np.min(plane_vertices, axis=0).tolist())
plane_faces_accessor = pygltflib.Accessor(bufferView=1, byteOffset=0, componentType=pygltflib.ComponentType.UNSIGNED_INT, count=len(plane_faces) * 3,
                                         type=pygltflib.Type.SCALAR)

glb.accessors.append(plane_vertices_accessor)
glb.accessors.append(plane_faces_accessor)

# Create a scene with the plane mesh
scene = pygltflib.Scene(nodes=[0])
glb.scenes.append(scene)
glb.scene = 0

output_folder = "/path/to/your/output/folder"

# Save the glTF scene as .glb file
glb.save(os.path.join(output_folder, 'output.glb'))
