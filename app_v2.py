import os
# --- CONFIGURACIÓN DE ENTORNO ---
os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
os.environ["TMPDIR"] = "/nas/antoniodetoro/qwen/tmp"
os.environ["PYTHONNOUSERSITE"] = "1"

import gradio as gr
import numpy as np
import random
import torch
import spaces
import tempfile
from PIL import Image
from typing import Optional, Tuple, Any

# Importaciones de Qwen (Asegúrate de tener estos paquetes en tu entorno)
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from gradio_client import Client, handle_file

# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargamos el transformer con low_cpu_mem_usage
transformer = QwenImageTransformer2DModel.from_pretrained(
    "linoyts/Qwen-Image-Edit-Rapid-AIO",
    subfolder='transformer',
    torch_dtype=dtype,
    low_cpu_mem_usage=True 
)

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=transformer,
    torch_dtype=dtype
)

# ACTIVAR OFFLOAD para CPUs (Crucial para GPUs de 24GB)
pipe.enable_sequential_cpu_offload() 

# Cargar pesos LoRA de ángulos
pipe.load_lora_weights(
    "dx8152/Qwen-Edit-2509-Multiple-angles",
    weight_name="镜头转换.safetensors",
    adapter_name="angles"
)

pipe.set_adapters(["angles"], adapter_weights=[1.])
pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
pipe.unload_lora_weights()

MAX_SEED = np.iinfo(np.int32).max

# --- Lógica de Prompts ---

def build_camera_prompt(camera_data: dict) -> str:
    """Convierte los datos del visualizador 3D en un prompt de texto."""
    if not camera_data:
        return "no camera movement"
        
    rotate_deg = camera_data.get("rotate_deg", 0)
    move_forward = camera_data.get("move_forward", 0)
    vertical_tilt = camera_data.get("vertical_tilt", 0)
    wideangle = camera_data.get("wideangle", False)
    
    prompt_parts = []

    if rotate_deg != 0:
        direction = "left" if rotate_deg > 0 else "right"
        if direction == "left":
            prompt_parts.append(f"将镜头向左旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the left.")
        else:
            prompt_parts.append(f"将镜头向右旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the right.")

    if move_forward > 5:
        prompt_parts.append("将镜头转为特写镜头 Turn the camera to a close-up.")
    elif move_forward >= 1:
        prompt_parts.append("将镜头向前移动 Move the camera forward.")

    if vertical_tilt >= 1:
        prompt_parts.append("将相机转向鸟瞰视角 Turn the camera to a bird's-eye view.")
    elif vertical_tilt <= -1:
        prompt_parts.append("将相机切换到仰视视角 Turn the camera to a worm's-eye view.")

    if wideangle:
        prompt_parts.append("将镜头转为广角镜头 Turn the camera to a wide-angle lens.")

    final_prompt = " ".join(prompt_parts).strip()
    return final_prompt if final_prompt else "no camera movement"

@spaces.GPU
def infer_camera_edit(
    image: Image.Image,
    prompt: str,
    seed: int,
    randomize_seed: bool,
    guidance_scale: float,
    num_steps: int,
    width: int,
    height: int
) -> Tuple[Image.Image, int]:
    """Ejecuta la generación basada en el prompt final (editable)."""
    if image is None:
        raise gr.Error("Por favor, sube una imagen primero.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        image=[image.convert("RGB")],
        prompt=prompt,
        height=height if height != 0 else None,
        width=width if width != 0 else None,
        num_inference_steps=num_steps,
        generator=generator,
        true_cfg_scale=guidance_scale,
        num_images_per_prompt=1,
    ).images[0]

    return result, seed

def update_dimensions_on_upload(image: Optional[Image.Image]) -> Tuple[int, int]:
    if image is None: return 1024, 1024
    original_width, original_height = image.size
    if original_width > original_height:
        new_width = 1024
        new_height = int(1024 * (original_height / original_width))
    else:
        new_height = 1024
        new_width = int(1024 * (original_width / original_height))
    return (new_width // 8) * 8, (new_height // 8) * 8

# --- Componente 3D (Mantenemos tus constantes de JS/HTML) ---

CAMERA_3D_HTML_TEMPLATE = """
<div id="camera-control-wrapper" style="width: 100%; height: 400px; position: relative; background: #1a1a1a; border-radius: 12px; overflow: hidden;">
    <div id="prompt-overlay" style="position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); padding: 8px 16px; border-radius: 8px; font-family: monospace; font-size: 11px; color: #00ff88; white-space: nowrap; z-index: 10; max-width: 90%; overflow: hidden; text-overflow: ellipsis;"></div>
    <div id="control-legend" style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 8px 12px; border-radius: 8px; font-family: system-ui; font-size: 11px; color: #fff; z-index: 10;">
        <div style="margin-bottom: 4px;"><span style="color: #00ff88;">●</span> Rotation (↔)</div>
        <div style="margin-bottom: 4px;"><span style="color: #ff69b4;">●</span> Vertical Tilt (↕)</div>
        <div><span style="color: #ffa500;">●</span> Distance/Zoom</div>
    </div>
</div>
"""


CAMERA_3D_JS = """
(() => {
    const wrapper = element.querySelector('#camera-control-wrapper');
    const promptOverlay = element.querySelector('#prompt-overlay');
    
    const initScene = () => {
        if (typeof THREE === 'undefined') {
            setTimeout(initScene, 100);
            return;
        }
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a);
        
        const camera = new THREE.PerspectiveCamera(50, wrapper.clientWidth / wrapper.clientHeight, 0.1, 1000);
        camera.position.set(4, 3, 4);
        camera.lookAt(0, 0.75, 0);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        wrapper.insertBefore(renderer.domElement, wrapper.firstChild);
        
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
        dirLight.position.set(5, 10, 5);
        scene.add(dirLight);
        
        scene.add(new THREE.GridHelper(6, 12, 0x333333, 0x222222));
        
        const CENTER = new THREE.Vector3(0, 0.75, 0);
        const BASE_DISTANCE = 2.0;
        const ROTATION_RADIUS = 2.2;
        const TILT_RADIUS = 1.6;
        
        let rotateDeg = props.value?.rotate_deg || 0;
        let moveForward = props.value?.move_forward || 0;
        let verticalTilt = props.value?.vertical_tilt || 0;
        let wideangle = props.value?.wideangle || false;
        
        const rotateSteps = [-90, -45, 0, 45, 90];
        const forwardSteps = [0, 5, 10];
        const tiltSteps = [-1, 0, 1];
        
        function snapToNearest(value, steps) {
            return steps.reduce((prev, curr) => Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev);
        }
        
        function createPlaceholderTexture() {
            const canvas = document.createElement('canvas');
            canvas.width = 256;
            canvas.height = 256;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#3a3a4a';
            ctx.fillRect(0, 0, 256, 256);
            ctx.fillStyle = '#ffcc99';
            ctx.beginPath();
            ctx.arc(128, 128, 80, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#333';
            ctx.beginPath();
            ctx.arc(100, 110, 10, 0, Math.PI * 2);
            ctx.arc(156, 110, 10, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(128, 130, 35, 0.2, Math.PI - 0.2);
            ctx.stroke();
            return new THREE.CanvasTexture(canvas);
        }
        
        let currentTexture = createPlaceholderTexture();
        const planeMaterial = new THREE.MeshBasicMaterial({ map: currentTexture, side: THREE.DoubleSide });
        let targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMaterial);
        targetPlane.position.copy(CENTER);
        scene.add(targetPlane);
        
        function updateTextureFromUrl(url) {
            if (!url) {
                planeMaterial.map = createPlaceholderTexture();
                planeMaterial.needsUpdate = true;
                scene.remove(targetPlane);
                targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMaterial);
                targetPlane.position.copy(CENTER);
                scene.add(targetPlane);
                return;
            }
            
            const loader = new THREE.TextureLoader();
            loader.crossOrigin = 'anonymous';
            loader.load(url, (texture) => {
                texture.minFilter = THREE.LinearFilter;
                texture.magFilter = THREE.LinearFilter;
                planeMaterial.map = texture;
                planeMaterial.needsUpdate = true;
                
                const img = texture.image;
                if (img && img.width && img.height) {
                    const aspect = img.width / img.height;
                    const maxSize = 1.4;
                    let planeWidth, planeHeight;
                    if (aspect > 1) {
                        planeWidth = maxSize;
                        planeHeight = maxSize / aspect;
                    } else {
                        planeHeight = maxSize;
                        planeWidth = maxSize * aspect;
                    }
                    scene.remove(targetPlane);
                    targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(planeWidth, planeHeight), planeMaterial);
                    targetPlane.position.copy(CENTER);
                    scene.add(targetPlane);
                }
            });
        }
        
        if (props.imageUrl) {
            updateTextureFromUrl(props.imageUrl);
        }
        
        const cameraGroup = new THREE.Group();
        const bodyMat = new THREE.MeshStandardMaterial({ color: 0x6699cc, metalness: 0.5, roughness: 0.3 });
        const body = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.2, 0.35), bodyMat);
        cameraGroup.add(body);
        const lens = new THREE.Mesh(
            new THREE.CylinderGeometry(0.08, 0.1, 0.16, 16),
            new THREE.MeshStandardMaterial({ color: 0x6699cc, metalness: 0.5, roughness: 0.3 })
        );
        lens.rotation.x = Math.PI / 2;
        lens.position.z = 0.24;
        cameraGroup.add(lens);
        scene.add(cameraGroup);
        
        const rotationArcPoints = [];
        for (let i = 0; i <= 32; i++) {
            const angle = THREE.MathUtils.degToRad(-90 + (180 * i / 32));
            rotationArcPoints.push(new THREE.Vector3(ROTATION_RADIUS * Math.sin(angle), 0.05, ROTATION_RADIUS * Math.cos(angle)));
        }
        const rotationCurve = new THREE.CatmullRomCurve3(rotationArcPoints);
        const rotationArc = new THREE.Mesh(
            new THREE.TubeGeometry(rotationCurve, 32, 0.035, 8, false),
            new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.3 })
        );
        scene.add(rotationArc);
        
        const rotationHandle = new THREE.Mesh(
            new THREE.SphereGeometry(0.16, 16, 16),
            new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.5 })
        );
        rotationHandle.userData.type = 'rotation';
        scene.add(rotationHandle);
        
        const tiltArcPoints = [];
        for (let i = 0; i <= 32; i++) {
            const angle = THREE.MathUtils.degToRad(-45 + (90 * i / 32));
            tiltArcPoints.push(new THREE.Vector3(-0.7, TILT_RADIUS * Math.sin(angle) + CENTER.y, TILT_RADIUS * Math.cos(angle)));
        }
        const tiltCurve = new THREE.CatmullRomCurve3(tiltArcPoints);
        const tiltArc = new THREE.Mesh(
            new THREE.TubeGeometry(tiltCurve, 32, 0.035, 8, false),
            new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xff69b4, emissiveIntensity: 0.3 })
        );
        scene.add(tiltArc);
        
        const tiltHandle = new THREE.Mesh(
            new THREE.SphereGeometry(0.16, 16, 16),
            new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xff69b4, emissiveIntensity: 0.5 })
        );
        tiltHandle.userData.type = 'tilt';
        scene.add(tiltHandle);
        
        const distanceLineGeo = new THREE.BufferGeometry();
        const distanceLine = new THREE.Line(distanceLineGeo, new THREE.LineBasicMaterial({ color: 0xffa500 }));
        scene.add(distanceLine);
        
        const distanceHandle = new THREE.Mesh(
            new THREE.SphereGeometry(0.16, 16, 16),
            new THREE.MeshStandardMaterial({ color: 0xffa500, emissive: 0xffa500, emissiveIntensity: 0.5 })
        );
        distanceHandle.userData.type = 'distance';
        scene.add(distanceHandle);
        
        function buildPromptText(rot, fwd, tilt, wide) {
            const parts = [];
            if (rot !== 0) {
                const dir = rot > 0 ? 'left' : 'right';
                parts.push('Rotate ' + Math.abs(rot) + '° ' + dir);
            }
            if (fwd > 5) parts.push('Close-up');
            else if (fwd >= 1) parts.push('Move forward');
            if (tilt >= 1) parts.push("Bird's-eye");
            else if (tilt <= -1) parts.push("Worm's-eye");
            if (wide) parts.push('Wide-angle');
            return parts.length > 0 ? parts.join(' • ') : 'No camera movement';
        }
        
        function updatePositions() {
            const rotRad = THREE.MathUtils.degToRad(-rotateDeg);
            const distance = BASE_DISTANCE - (moveForward / 10) * 1.0;
            const tiltAngle = verticalTilt * 35;
            const tiltRad = THREE.MathUtils.degToRad(tiltAngle);
            
            const camX = distance * Math.sin(rotRad) * Math.cos(tiltRad);
            const camY = distance * Math.sin(tiltRad) + CENTER.y;
            const camZ = distance * Math.cos(rotRad) * Math.cos(tiltRad);
            
            cameraGroup.position.set(camX, camY, camZ);
            cameraGroup.lookAt(CENTER);
            
            rotationHandle.position.set(ROTATION_RADIUS * Math.sin(rotRad), 0.05, ROTATION_RADIUS * Math.cos(rotRad));
            
            const tiltHandleAngle = THREE.MathUtils.degToRad(tiltAngle);
            tiltHandle.position.set(-0.7, TILT_RADIUS * Math.sin(tiltHandleAngle) + CENTER.y, TILT_RADIUS * Math.cos(tiltHandleAngle));
            
            const handleDist = distance - 0.4;
            distanceHandle.position.set(
                handleDist * Math.sin(rotRad) * Math.cos(tiltRad),
                handleDist * Math.sin(tiltRad) + CENTER.y,
                handleDist * Math.cos(rotRad) * Math.cos(tiltRad)
            );
            distanceLineGeo.setFromPoints([cameraGroup.position.clone(), CENTER.clone()]);
            
            promptOverlay.textContent = buildPromptText(rotateDeg, moveForward, verticalTilt, wideangle);
        }
        
        function updatePropsAndTrigger() {
            const rotSnap = snapToNearest(rotateDeg, rotateSteps);
            const fwdSnap = snapToNearest(moveForward, forwardSteps);
            const tiltSnap = snapToNearest(verticalTilt, tiltSteps);
            
            props.value = { rotate_deg: rotSnap, move_forward: fwdSnap, vertical_tilt: tiltSnap, wideangle: wideangle };
            trigger('change', props.value);
        }
        
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        let isDragging = false;
        let dragTarget = null;
        let dragStartMouse = new THREE.Vector2();
        let dragStartForward = 0;
        const intersection = new THREE.Vector3();
        
        const canvas = renderer.domElement;
        
        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects([rotationHandle, tiltHandle, distanceHandle]);
            
            if (intersects.length > 0) {
                isDragging = true;
                dragTarget = intersects[0].object;
                dragTarget.material.emissiveIntensity = 1.0;
                dragTarget.scale.setScalar(1.3);
                dragStartMouse.copy(mouse);
                dragStartForward = moveForward;
                canvas.style.cursor = 'grabbing';
            }
        });
        
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
            
            if (isDragging && dragTarget) {
                raycaster.setFromCamera(mouse, camera);
                
                if (dragTarget.userData.type === 'rotation') {
                    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
                    if (raycaster.ray.intersectPlane(plane, intersection)) {
                        let angle = THREE.MathUtils.radToDeg(Math.atan2(intersection.x, intersection.z));
                        rotateDeg = THREE.MathUtils.clamp(-angle, -90, 90);
                    }
                } else if (dragTarget.userData.type === 'tilt') {
                    const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0.7);
                    if (raycaster.ray.intersectPlane(plane, intersection)) {
                        const relY = intersection.y - CENTER.y;
                        const relZ = intersection.z;
                        const angle = THREE.MathUtils.radToDeg(Math.atan2(relY, relZ));
                        verticalTilt = THREE.MathUtils.clamp(angle / 35, -1, 1);
                    }
                } else if (dragTarget.userData.type === 'distance') {
                    const deltaY = mouse.y - dragStartMouse.y;
                    moveForward = THREE.MathUtils.clamp(dragStartForward + deltaY * 12, 0, 10);
                }
                updatePositions();
            } else {
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects([rotationHandle, tiltHandle, distanceHandle]);
                [rotationHandle, tiltHandle, distanceHandle].forEach(h => {
                    h.material.emissiveIntensity = 0.5;
                    h.scale.setScalar(1);
                });
                if (intersects.length > 0) {
                    intersects[0].object.material.emissiveIntensity = 0.8;
                    intersects[0].object.scale.setScalar(1.1);
                    canvas.style.cursor = 'grab';
                } else {
                    canvas.style.cursor = 'default';
                }
            }
        });
        
        const onMouseUp = () => {
            if (dragTarget) {
                dragTarget.material.emissiveIntensity = 0.5;
                dragTarget.scale.setScalar(1);
                
                const targetRot = snapToNearest(rotateDeg, rotateSteps);
                const targetFwd = snapToNearest(moveForward, forwardSteps);
                const targetTilt = snapToNearest(verticalTilt, tiltSteps);
                
                const startRot = rotateDeg, startFwd = moveForward, startTilt = verticalTilt;
                const startTime = Date.now();
                
                function animateSnap() {
                    const t = Math.min((Date.now() - startTime) / 200, 1);
                    const ease = 1 - Math.pow(1 - t, 3);
                    
                    rotateDeg = startRot + (targetRot - startRot) * ease;
                    moveForward = startFwd + (targetFwd - startFwd) * ease;
                    verticalTilt = startTilt + (targetTilt - startTilt) * ease;
                    
                    updatePositions();
                    if (t < 1) requestAnimationFrame(animateSnap);
                    else updatePropsAndTrigger();
                }
                animateSnap();
            }
            isDragging = false;
            dragTarget = null;
            canvas.style.cursor = 'default';
        };
        
        canvas.addEventListener('mouseup', onMouseUp);
        canvas.addEventListener('mouseleave', onMouseUp);

        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects([rotationHandle, tiltHandle, distanceHandle]);
            
            if (intersects.length > 0) {
                isDragging = true;
                dragTarget = intersects[0].object;
                dragTarget.material.emissiveIntensity = 1.0;
                dragTarget.scale.setScalar(1.3);
                dragStartMouse.copy(mouse);
                dragStartForward = moveForward;
            }
        }, { passive: false });
        
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;
            
            if (isDragging && dragTarget) {
                raycaster.setFromCamera(mouse, camera);
                
                if (dragTarget.userData.type === 'rotation') {
                    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
                    if (raycaster.ray.intersectPlane(plane, intersection)) {
                        let angle = THREE.MathUtils.radToDeg(Math.atan2(intersection.x, intersection.z));
                        rotateDeg = THREE.MathUtils.clamp(-angle, -90, 90);
                    }
                } else if (dragTarget.userData.type === 'tilt') {
                    const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0.7);
                    if (raycaster.ray.intersectPlane(plane, intersection)) {
                        const relY = intersection.y - CENTER.y;
                        const relZ = intersection.z;
                        const angle = THREE.MathUtils.radToDeg(Math.atan2(relY, relZ));
                        verticalTilt = THREE.MathUtils.clamp(angle / 35, -1, 1);
                    }
                } else if (dragTarget.userData.type === 'distance') {
                    const deltaY = mouse.y - dragStartMouse.y;
                    moveForward = THREE.MathUtils.clamp(dragStartForward + deltaY * 12, 0, 10);
                }
                updatePositions();
            }
        }, { passive: false });
        
        canvas.addEventListener('touchend', (e) => { e.preventDefault(); onMouseUp(); }, { passive: false });
        canvas.addEventListener('touchcancel', (e) => { e.preventDefault(); onMouseUp(); }, { passive: false });
        
        updatePositions();
        
        function render() {
            requestAnimationFrame(render);
            renderer.render(scene, camera);
        }
        render();
        
        new ResizeObserver(() => {
            camera.aspect = wrapper.clientWidth / wrapper.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
        }).observe(wrapper);
        
        wrapper._updateTexture = updateTextureFromUrl;
        
        let lastImageUrl = props.imageUrl;
        let lastValue = JSON.stringify(props.value);
        setInterval(() => {
            if (props.imageUrl !== lastImageUrl) {
                lastImageUrl = props.imageUrl;
                updateTextureFromUrl(props.imageUrl);
            }
            const currentValue = JSON.stringify(props.value);
            if (currentValue !== lastValue) {
                lastValue = currentValue;
                if (props.value && typeof props.value === 'object') {
                    rotateDeg = props.value.rotate_deg ?? rotateDeg;
                    moveForward = props.value.move_forward ?? moveForward;
                    verticalTilt = props.value.vertical_tilt ?? verticalTilt;
                    wideangle = props.value.wideangle ?? wideangle;
                    updatePositions();
                }
            }
        }, 100);
    };
    
    initScene();
})();
"""
# Se asume que el JS es el mismo que proveíste (initScene, THREE.js, etc.)
# Para ahorrar espacio en el script, incluimos la función de creación del componente:

def create_camera_3d_component(value=None, imageUrl=None, **kwargs):
    if value is None:
        value = {"rotate_deg": 0, "move_forward": 0, "vertical_tilt": 0, "wideangle": False}
    from qwen_js_code import CAMERA_3D_JS  # O pega aquí tu CAMERA_3D_JS directamente
    return gr.HTML(
        value=value,
        html_template=CAMERA_3D_HTML_TEMPLATE,
        js_on_load=CAMERA_3D_JS, # Pegar aquí el bloque CAMERA_3D_JS si no está externo
        imageUrl=imageUrl,
        **kwargs
    )

# --- UI CSS ---
css = '''
#col-container { max-width: 1100px; margin: 0 auto; }
.dark .progress-text { color: white !important; }
#camera-3d-control { min-height: 400px; }
'''

# --- BLOQUE PRINCIPAL DE GRADIO ---

with gr.Blocks(css=css, theme=gr.themes.Citrus()) as demo:
    gr.Markdown("## 🎬 Qwen Image Edit — Control de Cámara Manual")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="1. Imagen de Origen", type="pil")
            
            gr.Markdown("### 🎮 2. Ajustar Ángulo")
            # Usamos el componente HTML que maneja el 3D
            camera_3d = gr.HTML(
                value={"rotate_deg": 0, "move_forward": 0, "vertical_tilt": 0, "wideangle": False},
                html_template=CAMERA_3D_HTML_TEMPLATE,
                js_on_load=CAMERA_3D_JS, # Asegúrate de tener la variable CAMERA_3D_JS definida arriba
                elem_id="camera-3d-control"
            )
            
            prompt_display = gr.Textbox(
                label="3. Prompt Resultante (Puedes editarlo libremente)",
                placeholder="El prompt aparecerá aquí al mover el visor 3D...",
                lines=3
            )
            
            with gr.Row():
                reset_btn = gr.Button("🔄 Reset")
                run_btn = gr.Button("🚀 Confirmar y Generar", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_image = gr.Image(label="Imagen Generada", interactive=False)
            
            with gr.Accordion("Configuración Avanzada", open=False):
                seed_val = gr.Number(label="Seed", value=0, precision=0)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=5.0, value=1.0, step=0.1)
                num_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=20, value=4, step=1)
                width_val = gr.Number(label="Width", value=1024)
                height_val = gr.Number(label="Height", value=1024)

    # --- LÓGICA DE EVENTOS ---

    # Al cargar imagen, actualizar dimensiones
    input_image.upload(
        fn=update_dimensions_on_upload,
        inputs=[input_image],
        outputs=[width_val, height_val]
    )

    # CUANDO CAMBIA EL VISOR 3D: Solo actualizamos el prompt de texto, NO generamos.
    # Nota: El componente gr.HTML requiere que el JS use `gradio_element.dispatchEvent(new CustomEvent('change', ...))` 
    # para que Gradio detecte el cambio de valor.
    camera_3d.change(
        fn=build_camera_prompt,
        inputs=[camera_3d],
        outputs=[prompt_display]
    )

    # EL BOTÓN DE GENERAR: Toma el texto del prompt (que pudo ser editado) y genera.
    run_btn.click(
        fn=infer_camera_edit,
        inputs=[
            input_image, 
            prompt_display, 
            seed_val, 
            randomize_seed, 
            guidance_scale, 
            num_steps, 
            width_val, 
            height_val
        ],
        outputs=[result_image, seed_val]
    )

    # Reset
    reset_btn.click(
        fn=lambda: ({"rotate_deg": 0, "move_forward": 0, "vertical_tilt": 0, "wideangle": False}, ""),
        outputs=[camera_3d, prompt_display]
    )

if __name__ == "__main__":
    head = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
    demo.launch(share=True, head=head, footer_links=["api", "gradio", "settings"])